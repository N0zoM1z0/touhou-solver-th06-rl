# touhou-solver-th06-rl

A Wine-only research agent for original Japanese TH06 1.02h, aiming at a
complete Lunatic NMNB route and a reusable TH06-to-TH08 offline-RL method.

The architecture is deliberately small:

1. capture coherent factual Wine episodes;
2. filter actions against already-instantiated physical hazards with a bounded
   observed-hazard shield;
3. train offline and export an immutable lightweight policy.

It is **capture-complete, not prediction-complete**. ECL, stage, RNG, boss, and
spell facts may be stored for audit, but the online runtime does not interpret
future patterns and the actor may not depend on those identifiers. Unknown
future dynamics are a learning problem. The shield truthfully certifies only
its observed-object scope.

Start with [START_HERE.md](START_HERE.md). The research questions, method, and
evaluation ledger live under [paper/](paper/README.md). The exact online/offline
boundary is [docs/ONLINE_OFFLINE_SAFETY_CONTRACT.md](docs/ONLINE_OFFLINE_SAFETY_CONTRACT.md),
and the reusable corpus contract is
[docs/IMMUTABLE_WINE_DATA_PLANE.md](docs/IMMUTABLE_WINE_DATA_PLANE.md).

Portable Wine setup is documented in
[docs/PORTABLE_WINE_RUNTIME.md](docs/PORTABLE_WINE_RUNTIME.md). Extracted retail
stage scripts are optional research references described by
[docs/ECL_REFERENCE_CACHE.md](docs/ECL_REFERENCE_CACHE.md); they are not an
online dependency.

The first complete six-stage Lunatic baseline passed at commit
`af9900524520b72934a4c55e2f44118f88094633`: 108 HITs were retained through
Ending, with zero Bomb or infrastructure failure and an audited 135,561-frame
episode. The current milestone is the strict serial/parallel Wine differential,
followed by independent complete-route collection and the preregistered minimal
behavior-cloning learnability test.
