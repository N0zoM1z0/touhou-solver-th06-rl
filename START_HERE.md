# Start here

Read in this order:

1. `AGENTS.md` — non-negotiable project rules;
2. `paper/main.tex` — research questions, method, and experiment ledger;
3. `docs/ONLINE_OFFLINE_SAFETY_CONTRACT.md` — exact responsibility boundary;
4. `docs/IMMUTABLE_WINE_DATA_PLANE.md` — reusable factual episode contract;
5. `docs/PORTABLE_WINE_RUNTIME.md` — machine-independent Wine setup;
6. `docs/WINE_RETAIL_VALIDATION.md` — original-game runner details.

E0 is complete at commit
`af9900524520b72934a4c55e2f44118f88094633`: Practice Stages 4--6 and one
natural-RNG six-stage Lunatic route all completed and passed their factual
audits. The full route retained 108 HITs and zero Bomb.

Current checkpoint:

1. keep the audited E0 recorder/shield and corpus schema frozen;
2. retain the negative fixed-seed serial/parallel differential from commit
   `76782a4f37e0d12a9a2384561b53e68ceaf998ae`: every run passed its own audit,
   but exact factual/HIT equality failed, so no parallel pool is admitted;
3. implement only E2/L0 next: derive and audit causal decision-epoch learner
   rows from complete factual episodes, preserving command/sample/execution,
   behavior probabilities, exclusions, elapsed frames, and HIT conservation;
4. do not collect a new corpus, fit a model, widen `PolicyContext`, or run a
   learned Wine candidate until that data contract and its acceptance tests
   pass. Parallel collection remains disabled.

There is no authorized learned candidate yet. A complete route with many HITs
is still a successful infrastructure baseline when the factual and control
contracts pass.

The research goal is complete-route Lunatic NMNB probability. The first fitted
optimization target is expected undiscounted physical HIT count over complete
episodes; it is a dense surrogate and is not claimed equivalent to the goal
metric. The active learner order is causal data audit, then a frozen serial
inventory, action-frequency/reactive controls,
current-observation behavior cloning, one short-history ablation, and only then
one offline value method. Survival critics, IQL/CQL
sweeps, object/recurrent encoders, ensembles, active collection, learned MPC,
and Wine branching are not starting points.

After E2/L0 passes, the first training inventory is collected serially with
unchanged retail clock/update semantics, coherent capture suspension, the
frozen declared 20% uniform observed-shield mixture, and complete-episode
admission. Its episode count, seeds, stopping rule, and
whole-episode splits must be frozen before collection or fitting. This initial
inventory is not uncertainty-guided or adaptive collection.

Before reconsidering parallel or exact-prefix branch collection, investigate
and pass the generic coherent-root/action-delivery determinism boundary. Do not
relax exact equality or edit gameplay logic in response to a failure location.
