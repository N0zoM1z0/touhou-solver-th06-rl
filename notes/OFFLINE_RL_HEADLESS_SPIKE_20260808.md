# Sequential offline RL and Linux headless spike, 2026-08-08

This report records two acceleration experiments. Neither changes the online
authority boundary: learned decisions rank only a native-safe action set, Bomb
is not representable, and the shipped Windows game remains final physical-run
evidence.

## Sequential corpus contract

`src/th06_rl/offline_rl.py` exposes one-step MDP tuples from complete physical
Stages. A physical HIT penalty is attached once to the latest eligible action.
Bootstrapping stops at a HIT, Stage end, source-context change, incoherent next
snapshot, or an observation/control gap. It never crosses a missing or
ineligible decision.

At fixed dataset revision
`211374189e2ec074dafe817d529fd3de833dafd4`, the exact-v5 Lunatic/Reimu-A/Stage-6
scope contains three eligible complete Stages and 85,163 sequential rows:

- 84,258 rows (98.9373%) have a coherent one-step successor;
- terminals are 783 source-context boundaries, 93 observation/control gaps,
  26 physical HITs, and three Stage ends;
- 144 automatically derived source contexts are present; and
- every successor's logged action belongs to its recorded native legal set.

This is enough for conservative batch RL. It is not enough for a reliable
long-horizon policy-value claim: there is one held-out Stage, only 26 HIT
terminals, and long products of behavior propensities are unstable.

## CPU fitted-Q iteration

`scripts/train_offline_fqi.py` implements 30-step fitted-Q iteration with
per-physical-frame gamma 0.99. The `support` backup is a batch-constrained
variant: continuation maximization keeps the recorded reactive baseline and
actions with at least 32 observations in the same source context. Policy
extraction independently retains support-32 and a 0.5 baseline margin.

The first two chronological Stages (56,533 rows) train the model and the last
Stage (28,630 rows) is held out. Regressors ran serially at nice 15, at most 12
threads, and at most about 1.0 GiB process RSS.

| Backup / regressor | Held-out Bellman MAE | HIT-120 AP / ROC-AUC from `-Q(s,a_logged)` | Logged-action match | Seconds / peak RSS |
| --- | ---: | ---: | ---: | ---: |
| support / Extra Trees | 8.697 | 0.246 / 0.658 | 0.390 | 44.6 / 1,014.9 MiB |
| support / LightGBM | 9.553 | 0.239 / 0.666 | 0.495 | 29.6 / 757.9 MiB |
| support / XGBoost | 9.471 | 0.183 / 0.687 | 0.513 | 25.9 / 1,015.0 MiB |
| max / LightGBM | 9.410 | 0.238 / 0.672 | 0.499 | 33.0 / 760.9 MiB |

Every extracted policy selected zero Bomb actions and zero actions outside the
native-safe set. Support and unconstrained LightGBM are too close and disagree
by metric; the held-out residual is a Bellman-consistency diagnostic, not a
counterfactual return estimate. There is no offline promotion winner.

The next neural algorithms should be discrete CQL and IQL, still masked by the
recorded native-safe set. First collect at least ten clean v5 Stages per scope
and hold out at least three; otherwise a larger model mostly amplifies the same
support and terminal-sparsity problem.

## Authoritative-source and portable evidence

- The ignored authoritative checkout is `reference/GensokyoClub-th06` at
  `cc475a0bc3fef38683b0f02224c87ddba0a021d9`. Its `config/stubbed.csv` is empty
  and `config/implemented.csv` lists 503 reconstructed functions.
- The official portable branch was checked out as the separate ignored
  worktree `reference/GensokyoClub-th06-portable`, based on upstream commit
  `9a1c50b3e7821f2e32e0ff35de7e618216d796e5`.
- Local branch `th06-rl-headless-spike` has commits `0846d41` (Ubuntu 22.04 SDL
  compatibility), `8c30fbe` (deterministic logic-headless mode), and `cb60bf6`
  (direct Practice action/step protocol), `d0f92aa` (physical tick progression
  and fail-close terminal behavior), `4501c65` (public-fork boundary), and
  `294a478` (public usage and evidence documentation), `6e84840`
  (collision-complete observation v2), `d5e1737` (single-threaded headless
  runtime), `a87ada1` (stage-entry COW forkserver), and `034152c` (replayed
  checkpoints and terminal-only branches).
  `4b8c90b` then adds raw timeline clock/next-instruction identity for automatic
  offline partitioning without interpreting it as movement logic.
- The release Linux ELF builds with SDL2/image/ttf and Premake 5 beta2. A
  no-display, no-asset fail-close smoke exits cleanly three times with about
  22.9 MiB RSS and no leftover process or config write.

Headless mode forces SDL dummy video/audio, disables sound, skips the draw
chain, framebuffer/depth clears, swaps, and wall-clock pacing, and supports a
fixed 16-bit seed and tick bound. Direct Practice selection is parameterized by
stage, difficulty, character, and shot type. It is a generic state transition,
not a phase or boss script.

The 18-action input vocabulary matches the solver (`stay`, eight directions,
and `_fast` variants); optional Shoot is separate and Bomb is impossible to
encode. A run-length action stream can drive deterministic replay. `--step`
instead provides observation/action lockstep over stdout/stdin. JSON
observations include RNG state, player state, bullets, lasers, enemies, ECL
subroutine identity, lives, resources, and score.

`scripts/compare_headless_traces.py` reports the first structural or numeric
divergence path and tick. The real equivalence gate will compare identical DAT,
scope, seed, and action streams, first between repeated Linux runs and then
against a shipped-game exporter. Exact equality is attempted first; any float
tolerance must be explicit and justified.

## Real Japanese-data result

The uploaded RAR is 266,004,163 bytes with SHA-256
`6b013b24c101ae846b97a2778abf461d537640611a835824a42533c692be55d6`.
`unrar` verified every CRC. All 74 paths are contained below `th06/`; the
archive and extracted 331 MiB game tree remain below ignored `reference/` and
are not in either Git index. The six correctly named Japanese DAT files were
hashed separately. The built Linux ELF SHA-256 is
`961544485d5a788ef367ba9732b2d5dded8e2ff3bcb4fdda524a028004159c63`.

The first real test exposed and fixed a zero-time bug: the ordinary Render path
sets both framerate multipliers to one, while the initial headless fast path
left them zero. After the fix, direct Practice advances physical game time and
actions have their shipped effect:

- Stages 1 through 6 all directly start and advance 300 ticks under the same
  generic CLI; no menu automation or stage-specific movement branch is used.
- In Lunatic/Reimu-A/Stage 6, `right_fast` moves player x from 192 to 376 and
  focused Left returns it to 138. At tick 600 the run has 198 active bullets,
  six enemies, score 16,830, and RNG generation count 2,269.
- Two independent 600-tick processes with seed 7 and the same action stream
  produce byte-identical 1,313,466-byte traces. Both SHA-256 values are
  `c2208b923ea0a397fb7f04deef92f3b2e15a20356f6842036e1f3d7f5c56b2a0`.
  Seed 8 diverges at tick 1 in the recorded RNG state, as expected.
- Holding position reaches the first physical HIT at tick 848 with 368 active
  bullets. The default episode emits `physical-hit` and exits immediately.
- An action named `bomb` is rejected as unknown/forbidden, emits `input-error`,
  exits status 1, and leaves no process. A one-action file similarly fails
  closed on exhaustion before another player calc tick.
- The no-PTY Python client completes observation/action lockstep against the
  real game: consecutive `right_fast` actions move x 192 to 196 to 200; Left
  moves it back to 198; tick-limit is an explicit terminal observation.

At nice 15 and low I/O priority, a 5,000-tick continue-after-HIT workload takes
3.15 seconds without trace: 1,587 logic ticks/s, or 26.5 times real time. The
same workload with a 61,621,511-byte exported-hazard JSONL state trace takes 3.61
seconds: 1,385 ticks/s, or 23.1 times real time. Both stay below 38 MiB process
RSS. These are one-process measurements on a loaded shared VPS, not a claim
about multi-process scaling.

## COW counterfactual result

The Linux runtime now stays single threaded before a COW checkpoint: headless
sound initialization is skipped, and an actual loaded-DAT `/proc` audit found
one thread at the stage-entry boundary. A root server may fork one child to
replay any generic action prefix, stop at the requested physical tick, and then
fork serial short branches from that immutable state. `RUN_FINAL` writes only
the terminal observation so counterfactual labeling does not serialize every
intermediate hazard frame.

`scripts/benchmark_headless_branches.py` validates both speed and identity. In
Lunatic/Reimu-A/Stage 6, seed 7, a tick-600 checkpoint and all 18 constant-action
60-tick branches were compared against 18 cold full-prefix processes for three
repetitions:

- all 54 COW terminal observations matched their cold-replay counterpart byte
  for byte;
- each repetition found the same eight physical-HIT branches and ten branches
  surviving to tick 660;
- median COW time, including root startup and the one common prefix, was 0.2674
  seconds; median cold-replay time was 2.0932 seconds; and
- median per-repetition speedup was 7.87x on the loaded shared VPS.

The report is
`artifacts/benchmarks/headless-branches-stage6-seed7.json` and records source
commit `034152ca8f1635fbdc4d0e39dc6047a30d6d2e0c` plus binary SHA-256
`304c8aec5af50e1bb60c7c45522378c4e5722d9fcb6835eaae79a0beaa400f34`.
Artifacts remain ignored. This mechanism accelerates exact short-horizon
teacher labels, Q targets, and hard-example mining. It does not accelerate
gradient computation and has only modest benefit for a long whole-stage run.

## Compact headless corpus result

`scripts/generate_headless_corpus.py` drives the no-PTY step environment with a
12-frame offline native teacher plus exact epsilon exploration. The teacher
ranks only the current four-frame native-safe set. The same observation is
revalidated immediately before the synchronous action step, Bomb is absent
from the vocabulary, and the game still supplies the physical HIT terminal.
Timeline/boss identity is automatically derived only for data partitioning;
RNG state is retained by sparse source anchors but excluded from the compact
deployable state features.

Each compact row has a factual one-tick successor, canonical current/next
observation SHA-256, exact marginal behavior probability, native legal set,
per-candidate clearance/final-position/reserve data, teacher metadata, and
primitive outcome terms. Full observation-v2 roots are gzip anchors every 120
ticks and at initial/terminal/authority boundaries. Shards and their hashes are
published transactionally through a manifest.

Two real Lunatic/Reimu-A/Stage-6 seed-7 smoke trajectories were independently
re-audited by `scripts/audit_headless_corpus.py`:

- 1,798/1,798 rows have a consecutive factual successor, a selected native-
  legal Bomb-free action, a valid behavior probability, and zero Bomb delta;
- the digest chain and all 17 sparse anchors link back to the compact rows;
- the 600-tick run has 599 rows and 21 automatic contexts in 152,451 compressed
  bytes;
- the 1,200-tick run has 1,199 rows and 96 automatic contexts in 503,910
  compressed bytes; and
- the 12-frame teacher reached tick 1,200 without HIT or authority failure,
  passing the earlier generic fallback's tick-1,095 local dead end.

The combined independent report is
`artifacts/benchmarks/headless-corpus-audit.json`; generated data remains
ignored. These are truncated smoke runs, not a Stage-clear or NMNB claim. For
training support, collect multiple seeds and exploration rates per explicit
scope rather than treating one deterministic seed as an episode population.

The public source-only fork is
[`N0zoM1z0/th06-headless`](https://github.com/N0zoM1z0/th06-headless), with
default branch `main`. It contains no original executables, DAT, WAV, RAR,
corpus, model, or generated trace.

## Remaining equivalence gate

This is now a demonstrated deterministic Linux game environment, but it is not
yet a validated substitute for the shipped Windows binary. The remaining hard
gate is a Windows exporter that replays the same scope, seed, and per-frame
action stream and records the same physical contract. We must find and explain
the first divergence in RNG, player, bullets, lasers, enemies, HIT, or terminal
state. Capture latency, observation loss, uncertainty, and input-delivery noise
from the real agent must also be injected during training; headless source
state may not silently become a deployment-only feature.
