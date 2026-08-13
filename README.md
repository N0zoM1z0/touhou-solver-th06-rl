# touhou-solver-th06-rl

A Wine-only learning agent for original Japanese TH06 1.02h.

Original-retail Wine is the sole gameplay environment. The online loop captures
one coherent physical state, lets a native kernel certify the Bomb-free safe
first-action set, ranks only that set with an immutable lightweight policy,
freshly revalidates the selected action, and publishes it through an exact-PID
background input bridge.

The learning loop is deliberately asymmetric:

`Wine exploration -> grouped offline learning -> immutable candidate -> Wine canary`

Wine supplies every factual outcome. Offline jobs reuse Wine trajectories to
construct environment-neutral transitions, train grouped action-value
residuals, and shadow-score candidates. The resident policy performs no
learning and defaults to the frozen incumbent outside independently supported
regions. Policy quality is improved by unattended data rounds, never by
hand-tuning a failure location.

Corpus, learner/framework, and fitted result have independent identities and
lifecycles. A corpus is immutable reusable Wine evidence, not property of the
algorithm that first used it. Replacing or repairing a learner must replay all
compatible registered episodes before collecting more gameplay, so different
offline RL methods can be compared on the same facts without changing their
distribution. The normative contract is
[docs/IMMUTABLE_WINE_DATA_PLANE.md](docs/IMMUTABLE_WINE_DATA_PLANE.md).

The reconstructed Linux/headless simulator is retired from the learning and
evaluation path. Historical scripts remain only as unreferenced quarantine
until a separate cleanup removes code proven unused by the Wine-only path.

Start with [START_HERE.md](START_HERE.md). The authoritative method and
evaluation contract is
[docs/WINE_ONLY_AUTONOMOUS_LEARNING.md](docs/WINE_ONLY_AUTONOMOUS_LEARNING.md).
The original-retail runner contract is
[docs/WINE_RETAIL_VALIDATION.md](docs/WINE_RETAIL_VALIDATION.md).

Final policy evidence is the physical HIT count in alternating, normal-speed,
complete original-retail Wine Practice Stages with HIT continuation. Fixed RNG,
accelerated Wine, first-failure prefixes, shadow replay, and offline metrics may
reject or select a candidate, but may not promote one.

Run the current fully predeclared, resumable Generation-6 round with:

```bash
PYTHONPATH=.:src .venv/bin/python scripts/run_generation6_autonomous_round.py \
  --contract config/autonomous_generation6_round1.json \
  --output-root artifacts/autonomous-generation-6-round-1
```

The contract owns twelve balanced collection Stages, an all-registry grouped
refit, full Linux/Wine native smoke, bounded Wine canaries, and (only after
those gates pass) twelve alternating complete-Stage-6 A/B trials.
`generation.json` is the crash/audit ledger. Do not alter the contract after it
has been allowlisted; start a new round for a general algorithm change.
