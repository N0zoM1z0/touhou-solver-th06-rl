# touhou-solver-th06-rl

A Wine-only research agent for original Japanese TH06 1.02h, aiming at a
complete Lunatic NMNB route and a reusable TH06-to-TH08 learning method. Offline
RL is a candidate inside the research ladder, not a presupposed answer.

The architecture is deliberately small:

1. capture coherent factual Wine episodes;
2. filter actions against already-instantiated physical hazards with a bounded
   observed-hazard shield;
3. derive auditable learner data, fit offline, and export an immutable
   lightweight scorer that can only rank the shield set.

It is **capture-complete, not prediction-complete**. ECL, stage, RNG, boss, and
spell facts may be stored for audit, but the online runtime does not interpret
future patterns and the actor may not depend on those identifiers. Unknown
future dynamics are a learning problem. The shield truthfully certifies only
its observed-object scope.

The scientific goal metric is complete-route NMNB probability. The initial
optimization target is expected undiscounted physical HIT count over complete
episodes. HIT count is deliberately a dense surrogate, not a claim that it
orders every policy exactly like NMNB probability. No clearance, graze, score,
route-progress, stage, spell, ECL, or RNG reward is allowed.

The first learner is intentionally small: an exact decision-epoch data audit,
one frozen serial complete-episode training inventory, action-frequency/reactive
controls, and current-observation behavior cloning. One short-history ablation
and then one offline value method may follow only after the preceding gate
passes. Recurrent object encoders, survival critics, ensembles, active
collection, learned dynamics/MPC, and exact-prefix Wine branching are deferred
hypotheses rather than active architecture.

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
episode. The strict shared-host serial/parallel differential was then executed
at commit `76782a4f37e0d12a9a2384561b53e68ceaf998ae`. All three Stage 4 runs passed
their individual audits, but their fixed-seed trajectories and HIT counts
differed, so the exact gate correctly rejected the pool. Parallel collection is
disabled. This checkpoint stops at validated serial infrastructure and baseline
evidence; no corpus expansion or learning run was started. The next active work
is E2/L0 causal decision-epoch reconstruction and its tests. No fitted model or
learned Wine candidate is authorized until that contract passes.
