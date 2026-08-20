# Original-retail Wine validation

This path runs the original Japanese TH06 1.02h executable under Wine. Wine is
the project's only gameplay, learning, and evaluation environment. Fixed RNG,
accelerated Wine, and offline replay remain diagnostic/training strata; final
policy comparison uses normal-speed complete original-retail Wine Stages.

## Fixed inputs

The runner accepts only the original executable whose SHA-256 is
`9f76483c46256804792399296619c1274363c31cd8f1775fafb55106fb852245`.
The game, score file, Windows Python, native DLL, Wine prefix, and all run
artifacts are ignored and must not be committed.

The canonical local layout is:

```text
reference/th06-game-original/th06/                 original game directory
reference/th06-game-original/full-unlock-score.dat ignored unlock template
reference/tools/windows-python-3.11.9-embed-win32/ ignored 32-bit Python
reference/wine-prefixes/th06-retail/                ignored 32-bit Wine prefix
build/native-win32-fully-static/                    ignored native DLL build
```

The unlock template is restored before every run and must hash to
`54cd436d5d8a7a904190c792a977bf270ab1cb759fd72101e51e94d26b749c71`.
This makes all Practice stages selectable without committing the proprietary
save file.

## Source grounding

Every shipped-game assumption used by the adapter is traceable to the exact
GensokyoClub/th06 commit recorded in `RETAIL_ADAPTER_PROVENANCE.md`. The source
checkout may live anywhere and is read-only; setup can attest it with
`--exact-source` but runtime does not depend on it:

- `src/Supervisor.hpp` defines the 1.02 configuration layout, the windowed
  byte, and the Supervisor gameplay/ending states.
- `src/Supervisor.cpp`, `src/utils.hpp`, and `config/globals.csv` define the
  current/last input sampling and held-input repeat globals. The menu adapter
  waits for those physical globals instead of guessing a Wine sleep duration.
- `src/MainMenu.cpp` defines Start cursor 0, Practice cursor 2, the selection
  edge, and the startup refresh-rate timing loop.
- `src/Gui.cpp` sends an ordinary six-stage non-Practice route to the Ending
  Supervisor state.
- `src/Player.cpp` contains the original death, respawn, life decrement, and
  retry-menu behavior. The continuation patch changes only the life exhaustion
  branch; the original collision and HIT transition remain observable.

`scripts/configure_wine_retail.py` changes only the source-defined color-mode
and windowed configuration bytes. The GDB startup helper verifies the exact
1.02h instruction bytes, normalizes the Wine-only stale startup clock, emits a
required marker, and detaches before menu or battle control begins. It does not
patch bullet, laser, enemy, player-collision, or movement physics.

## Provision from a fresh checkout

The portable, idempotent path installs the host requirements, materializes all
ignored inputs, creates the source-derived windowed config, and builds the
controller DLL:

```bash
scripts/bootstrap_wine_runtime.sh
scripts/smoke_wine_runtime.sh
```

See `PORTABLE_WINE_RUNTIME.md` for inputs, overrides, and exact smoke gates.

## Build the controller DLL manually

The shipped executable and the embedded controller are 32-bit processes. Build
the native gate with MinGW and static compiler runtimes:

```bash
cmake -S native -B build/native-win32-fully-static \
  -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/mingw32.cmake \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/native-win32-fully-static -j4
```

Every run records the actual DLL SHA-256. Never infer compatibility from a
historical DLL identity.

## Run

Wine, Xvfb, GDB, a Japanese UTF-8 locale, and the ignored inputs above are
required. The game never runs through a PTY. The separate Windows console
controller uses a PTY bridge where Wine requires valid console handles, with
all output forwarded to its artifact log. The runner refuses a busy Wine prefix
or display, verifies every fixed hash, restores the unlock save, normalizes
startup, starts the Bomb-free controller, and cleans only its exact dedicated
prefix and processes.

Run a natural Practice Stage 6 continuation benchmark:

```bash
POLICY_PLUGIN=path/to/frozen_candidate.py \
POLICY_STATE=path/to/frozen_candidate.json \
.venv/bin/python scripts/run_wine_retail.py \
  --practice-stage 6 \
  --policy-plugin "$POLICY_PLUGIN" \
  --policy-state "$POLICY_STATE" \
  --immutable-policy \
  --artifact-dir artifacts/wine-retail-stage6-example
```

Run the ordinary Reimu-A Lunatic route from Start through Ending:

```bash
POLICY_PLUGIN=path/to/frozen_candidate.py \
POLICY_STATE=path/to/frozen_candidate.json \
.venv/bin/python scripts/run_wine_retail.py \
  --start-route \
  --policy-plugin "$POLICY_PLUGIN" \
  --policy-state "$POLICY_STATE" \
  --immutable-policy \
  --artifact-dir artifacts/wine-retail-route-example
```

Collect and audit the frozen native-baseline route before fitting a learner:

```bash
scripts/collect_baseline_route.sh
```

This is one natural-RNG physical episode spanning declared Stages 1--6. The
stage-reset transitions remain factual but are excluded from learning because
their frame gap is not one. Completion verifies HIT conservation across the
controller log, trace, manifest outcome, and lossless event stream.

`--seconds 0` means natural completion, not an infinite or arbitrary timeout.
HIT continuation is explicitly benchmark-only and training-ineligible. Bomb is
unrepresentable in the input bridge. Every artifact directory contains the
controller/game/GDB logs, a frame trace, configuration evidence, and
`report.json` with process cleanup and provenance.

The runner records the explicit policy plug-in and policy-state paths,
before/after hashes, and the exact controller command. Learning runs use an
immutable policy plus an explicit propensity-recorded randomized exploration
policy. Final comparisons require `--immutable-policy` and exploration zero.

## Evidence boundary

Passing this runner proves that the original retail executable accepted the
controller's background input, exposed coherent live state, counted physical
HITs, completed a Practice stage, or reached the original Ending state. It does
not prove policy quality by itself. Promotion follows the complete-Stage
alternating HIT-count contract in `WINE_ONLY_AUTONOMOUS_LEARNING.md`.
