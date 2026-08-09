# Exact-state offline feasibility benchmark

This benchmark separates three failure axes before another learner is trained:

1. **geometry / authority / search ceiling** -- no tested continuation finds a
   no-HIT, no-Bomb, authority-preserving branch for any native-safe first
   action;
2. **policy selection** -- at least one tested continuation constructs a
   feasible branch, but the factual policy's first action has no witness;
3. **representation / learner separation** -- a leave-one-seed-out probe
   compares the current compact candidate view with a bounded richer feature
   vector derived from the exact physical snapshot.

The benchmark is offline-only. It does not change the resident controller,
interpret future ECL births, add an action, weaken the native gate, or publish
Bomb. Search and richer source-physical features are diagnostics and may not
become collision authority.

## Epistemic boundary

The generator exhausts the declared Cartesian product of native-safe first
actions and continuation policies. A successful branch is a constructive
feasibility witness. Failure to find one is `oracle-no-witness`, **not** proof
that no action sequence exists. Increasing the continuation population or
branch horizon can turn a no-witness checkpoint into a witness; reports retain
the exact population, horizon, runtime hash, corpus hashes, and implementation
commit for that reason.

After an exhaustive pass, `--first-action` and `--continuation` may declare a
strict branch subset for a longer extension. Such an artifact is marked
`declared-subset`; it can preserve or refute a particular bounded witness but
cannot relabel untested actions negative. The auditor excludes subset results
from checkpoint-wide bottleneck rates and representation training.

Each continuation dynamically rebuilds the native safe set and repeats the
fresh issue check on every tick. A branch is feasible only when it reaches the
requested tick limit or a source-reported successful Stage/chain exit with
zero physical deaths, zero Bomb use, and no authority failure.

## Continuation population

The default population contains:

- the constant-time generic clearance/boundary fallback;
- native local planners at horizons 4, 12, 30, and 60;
- any explicitly supplied distilled rankers, each identified by SHA-256.

This removes the old single-teacher continuation assumption while retaining a
bounded and auditable experiment. Ranker artifacts must match the exact scope
and clean headless source build.

## Generate and audit

Select only reconstructable rows before the first physical HIT or benchmark
authority release. Existing event selection in `batch_label_headless_cow.py`
can be used to discover those sequences. For one run:

```bash
PYTHONPATH=.:src .venv/bin/python \
  scripts/label_headless_feasibility_oracle.py RUN \
  --checkpoint-sequence SEQUENCE \
  --branch-frames 1200 \
  --planner-horizon 4 --planner-horizon 12 \
  --planner-horizon 30 --planner-horizon 60 \
  --model artifacts/models/STAGE_INCUMBENT/teacher-ranker.joblib \
  --output artifacts/feasibility/STAGE-SEED.json

# Extend one bounded witness without rerunning the full Cartesian product.
PYTHONPATH=.:src .venv/bin/python \
  scripts/label_headless_feasibility_oracle.py RUN \
  --checkpoint-sequence SEQUENCE --branch-frames 1200 \
  --planner-horizon 12 --first-action stay \
  --continuation native-local-h12 \
  --output artifacts/feasibility/STAGE-SEED-extension.json

PYTHONPATH=.:src .venv/bin/python \
  scripts/audit_headless_feasibility_oracle.py \
  artifacts/feasibility/STAGE-SEED.json \
  --output artifacts/benchmarks/STAGE-feasibility-audit.json
```

Formal generation refuses a dirty repository by default. `--allow-dirty-code`
exists only for local smoke tests and is recorded as dirty provenance.

Geometry changes intentionally invalidate the compact corpus legal set even
when the source checkpoint remains byte-logically identical. For a declared
before/after authority experiment, add `--allow-native-set-revision`. The
artifact then records both `input_native_legal_actions` and the recomputed
`native_legal_actions`; the independent audit rejects a silent revision.
It also recomputes `runtime_compact_state` and `runtime_action_candidates` for
the revised set, while retaining the original corpus records separately. This
keeps newly exposed actions available to representation probes and later
offline supervision instead of silently dropping them from the feature table.

The runtime artifact also records `runtime_delivery_contract` and
`runtime_delivery_delays`. Linux `STEP` is synchronous (`[0]`); this must not be
silently mixed with an asynchronous Windows pickup envelope. Both remain
native-gated contracts rather than policy-controlled uncertainty settings.

Each newly generated branch records a run-length encoded action trace, its
SHA-256, terminal exact-observation fingerprint, tick, player position, and
hazard counts. This makes an authority failure reproducible as another exact
checkpoint instead of leaving only a vague final-frame label.

Replay one such closure and compare configured Hard, margin-zero Hard, and all
18 source-executed constant actions over the exact four-tick window with:

```bash
PYTHONPATH=.:src .venv/bin/python \
  scripts/audit_headless_authority_failure.py ORACLE.json \
  --checkpoint-sequence SEQUENCE \
  --first-action ACTION --continuation CONTINUATION \
  --output artifacts/benchmarks/authority-failure-differential.json
```

Agreement between margin-zero Hard and source execution isolates a configured
conservative-margin closure; disagreement is a geometry-model counterexample.
No source-safe constant action is only a four-tick constant-action dead end,
not a proof against a changing-action sequence or earlier avoidance.

The independent artifact audit recomputes branch feasibility,
action/continuation coverage, trace integrity, best-action summaries, policy
verdicts, Bomb deltas, bounds, and source identity. It rejects a missing branch
or a summary that does not match the recorded physical outcomes. Replaying a
terminal fingerprint in the authoritative runtime is a separate differential
experiment and must not be implied by the structural audit alone.

## Representation probe

Every exact checkpoint records two non-authoritative learning views:

- the same compact state and candidate records consumed by the resident
  ranker;
- a bounded nearest-hazard vector derived from the exact bullet, laser, enemy,
  and player snapshot.

For checkpoints where some but not all first actions have feasibility
witnesses, the auditor trains an Extra Trees diagnostic with leave-one-seed-out
splits. It reports the feasible-action top-1 rate for compact and exact-derived
views, plus the factual/local-teacher rate.

Interpret the deltas conservatively:

- high `oracle-no-witness` rate points first to native feasibility, observation
  authority, insufficient continuation coverage, or insufficient horizon;
- higher exact-derived than compact held-out rate is evidence of compact
  representation loss;
- higher compact-probe than factual-policy rate is evidence that the compact
  view contains usable signal which the current learner did not extract;
- too few seeds, too few discriminative checkpoints, or missing class support
  produces an explicit insufficient-evidence status rather than a verdict.

Only complete-seed, exact-scope comparisons may be combined. The final NMNB
gate remains a natural full-Stage headless clear followed by differential and
physical Windows validation.

Confirmed authority failures and source-physical geometry corrections are
recorded separately in
[`GEOMETRY_AUTHORITY_FINDINGS.md`](GEOMETRY_AUTHORITY_FINDINGS.md).
