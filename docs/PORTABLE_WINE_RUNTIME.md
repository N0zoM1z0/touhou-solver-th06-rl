# Portable Wine runtime

The repository can provision and smoke-test its original-retail runtime without
the historical `N0zoM1z0/touhou-solver-th06` checkout. All tracked paths are
derived from the current script location; game assets, Wine state, downloaded
tools, builds, and run artifacts remain under ignored directories.

## One-time setup

On Debian 12 (or a compatible apt-based host), place the legally obtained
`th06.rar` at the default sibling path `../game-exe/th06.rar`, then run:

```bash
scripts/bootstrap_wine_runtime.sh
```

The script enables i386 packages, installs Wine/Xvfb/GDB/MinGW and Python build
dependencies, creates `.venv`, installs this repository editable, extracts and
hash-attests TH06, downloads the pinned official Python 3.11.9 Win32 embeddable
runtime, generates its repo-relative import paths, creates the source-defined
windowed TH06 configuration, and builds both host and Win32 native libraries.
It is idempotent and refuses mismatched existing executable, score, or Python
identities.

Existing machines can avoid package installation. The exact-source checkout is
optional attestation input, not a runtime dependency:

```bash
scripts/bootstrap_wine_runtime.sh \
  --skip-system-packages \
  --exact-source ../th06
```

Use `--archive`, `--game-dir`, and `--wine-prefix` to override discovered
locations. No absolute workspace path is stored in tracked files. A custom
game location passed to the smoke runner also needs the matching
`--score-template`; `scripts/smoke_wine_runtime.sh` forwards runner options.

The original archive and extracted game are never added to Git. The accepted
executable and full-unlock save hashes are documented in
`WINE_RETAIL_VALIDATION.md`.

For parallel training collection, provision the ordinary runtime first, then
create the independent worker pool from the original archive:

```bash
.venv/bin/python scripts/prepare_wine_workers.py
```

This extracts a separate never-executed template, verifies the archive and
retail executable hashes, freezes its inventory, and creates two independent
game directories, Wine prefixes, displays, and CPU partitions. It initializes
both Wine prefixes without starting TH06, so the later serial/concurrent
differential compares equally warm prefixes. See `WINE_EXACT_ACCELERATION.md` for the required compatibility
gate and collector; canary/final runs never use the parallel collector.

## End-to-end verification

The following launches Lunatic Practice Stage 1 for a short bounded window:

```bash
scripts/smoke_wine_runtime.sh
```

The smoke uses an explicit immutable infrastructure policy that delegates to
the native reactive baseline. It is not a learned candidate, does not write a
corpus, and is not promotion evidence. Success requires all of these signals:

- exact retail identity and GDB startup normalization;
- Win32 Python importing this checkout without another solver repository;
- attachment to the exact TH06 process and coherent gameplay frames;
- at least one native-safe agent decision and physical input publication;
- immutable policy-state hashes and zero leftover prefix processes.

`TH06_RL_SMOKE_SECONDS` changes the bounded duration. The runner writes an
ignored artifact directory containing `report.json`, logs, and the live trace;
`scripts/verify_wine_smoke.py` fails closed if the decisive signals are absent.

The game process is launched with stdin detached and never receives a PTY.
Wine's Windows console Python requires valid console handles on some versions,
so the separate controller process uses a host PTY bridge whose full output is
forwarded into `controller.log`.

The GDB normalization currently runs through `sudo -n`; unattended use by a
non-root account therefore needs narrowly scoped passwordless sudo or an
equivalent host ptrace configuration.
