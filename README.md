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
