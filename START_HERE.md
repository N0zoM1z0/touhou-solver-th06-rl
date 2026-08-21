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
3. stop here. No additional route collection, train/validation/test inventory,
   behavior cloning, offline RL, or online learned candidate is authorized in
   this checkpoint.

There is no authorized learned candidate yet. A complete route with many HITs
is still a successful infrastructure baseline when the factual and control
contracts pass. If work resumes, investigate the generic capture/root boundary
before reconsidering parallel collection; do not relax exact equality or edit
gameplay logic in response to a failure location.
