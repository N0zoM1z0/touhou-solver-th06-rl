# CPU offline training and benchmark, 2026-08-08

This report freezes one reproducible experiment. It is an offline qualification
and experiment-ordering result, not authority to publish a learned action or to
replace the reactive baseline.

## Immutable inputs and host limits

- Hugging Face dataset: `Joh1rreq/touhou-solver-th06-rl-corpus`, revision
  `211374189e2ec074dafe817d529fd3de833dafd4`.
- Remote snapshot inventory: 14,926 files and 27,707,431,883 bytes. The compact
  local training view contains every run descriptor, manifest, event stream,
  and transition stream. Only nine full-frame shards were fetched for the
  recorded-frame benchmark.
- Authoritative source: ignored checkout `reference/GensokyoClub-th06` from
  [`GensokyoClub/th06`](https://github.com/GensokyoClub/th06), clean commit
  `cc475a0bc3fef38683b0f02224c87ddba0a021d9`. The native benchmark checks the
  playfield, player-hitbox, bullet EX-reflection, and collision-call anchors
  before running.
- Host: AMD EPYC 9654, no visible GPU. Formal model fits ran sequentially with
  at most 12 model threads, `nice -n 15`, and low-priority buffered I/O. The
  largest run peaked at 5,265.5 MiB RSS. (The effective nice value was 10
  because the invoking process inherited nice -5.)

The authoritative clone, corpus, models, virtual environment, and generated
reports are intentionally ignored by Git.

## Benchmark contract

| Boundary | Evidence | Result |
| --- | --- | --- |
| Geometry | Independent scalar AABB oracle against the native kernel | 4,000 cases, 0 mismatches; one-action P95 6.089 us |
| Native gate | Determinism, collision-margin monotonicity, and endpoint bounds | 800 cases, 0 failures; 18-action/H4 P95 47.742 us |
| Local planning | H12 plan must select from the already-certified Hard set and remain in bounds | 122 timed nonempty-Hard cases from 160 attempts, 0 failures; P95 1,559.076 us |
| Recorded decision path | v5 compact context versus the same physical frame root | 4,225 paired frames across 87 source contexts, 0 coherence mismatches and 0 Bomb observations |
| Recorded latency | Physical capture plus solve, including dense bullet frames | overall P95 8.110 ms; 500+ bullets P95 8.730 ms across 72 frames; overall over-16.67-ms rate 0.071% |
| Learning | Chronological split by complete physical Practice Stage; candidates restricted to recorded native-safe sets | all four model families and every policy variant selected 0 actions outside the safe set and 0 Bomb actions |

The recorded-frame result is a stratified sample from one latest held-out v5
Stage 6 run, not a scan of all 27.3 GB of frame shards. Its 72-frame 500+
bullet bin passes the budget but is still too small to claim a population tail
bound.

## Effective corpus

A factual training row must belong to an indexed training-eligible complete
Stage, be transition-learning-eligible, have proposal equal publication, have
the published action inside the recorded native legal set, carry a valid
behavior probability, and contain no Bomb outcome.

| Scope (difficulty/character/shot/stage) | Runs | Eligible complete Stages | Rows | Factual trainable | Exact v5 context within trainable | HIT-within-120 positive rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Hard / Reimu-A / Stage 4 (`2/0/0/4`) | 8 | 3 | 126,009 | 68,486 (54.35%) | 0 | 2.37% |
| Lunatic / Reimu-A / Stage 4 (`3/0/0/4`) | 5 | 5 | 130,729 | 129,664 (99.19%) | 0 | 1.20% |
| Lunatic / Reimu-A / Stage 5 (`3/0/0/5`) | 5 | 3 | 90,000 | 57,494 (63.88%) | 0 | 4.90% |
| Lunatic / Reimu-A / Stage 6 (`3/0/0/6`) | 77 | 76 | 2,384,986 | 2,287,269 (95.90%) | 85,163 (3.72%) | 3.59% |
| **All present scopes** | **95** | **87** | **2,731,724** | **2,542,913 (93.09%)** | **85,163 (3.35%)** | **3.46%** |

The full SHA-256, gzip/JSON schema, row-count, and sequence audit passed. There
were 0 Bomb rows, 2 authority-loss rows, 29,419 control-dead-end rows, 11,324
observation-gap rows, 19,026 proposal/publication mismatches, and 1,085 physical
HIT events. The snapshot contains no Stage 1-3 runs even though the current
collection target covers Stages 1-6.

All 18 movement actions have factual support, but the behavior distribution is
highly concentrated: 2,656,962 of 2,731,724 rows have propensity in `[0.8,1]`,
while only 74,762 are below 0.05. Stage 5 is the weakest active scope: its
minimum per-action clipped IPW effective sample size is 259, versus 9,544 for
Stage 6.

## CPU policy zoo

`common` uses fields faithfully present in all transition schemas. Missing v4
compact-context values stay explicitly unknown; they are not invented from
future state. `exact-v5` uses the small context available to the online policy
contract. Every split holds out the latest complete physical Stage by run ID.

| Scope/view | Train Stages / rows | Held-out Stage rows | Best factual MAE | Best HIT-120 ROC-AUC | Peak RSS |
| --- | ---: | ---: | --- | --- | ---: |
| Stage 4 common | 4 / 103,280 | 26,384 | XGBoost 0.552 | Extra Trees 0.954 | 1.71 GiB |
| Stage 5 common | 2 / 38,522 | 18,972 | Extra Trees 2.164 | Extra Trees 0.884 | 0.97 GiB |
| Stage 6 common | 75 / 500,000 (from 2,258,639) | 28,630 | CatBoost 1.254 | Extra Trees 0.917 | 5.14 GiB |
| Stage 6 exact-v5 | 2 / 56,533 | 28,630 | Extra Trees 1.114 | Extra Trees 0.900 | 1.32 GiB |

The Stage 6 cap uses a deterministic reservoir for ordinary rows, retains all
HIT/failure windows, and records inclusion weights (maximum 5.218). Training
also tempers inverse propensity by a clipped square root. This keeps rare
exploration useful without allowing it to dominate a small physical split.

Counterfactual estimates do not give a promotion winner:

- Unconstrained argmax policies frequently have low effective sample size and
  contradictory DR/WIS signs. They should not be tested physically first.
- On Stage 6 common, support-32/margin-0.5 variants retain ESS 15,620-16,648
  (reactive baseline ESS 17,193), but model-dependent DR and WIS do not agree on
  one family. Extra Trees reaches WIS 0.932 versus baseline 0.909 but only a
  small DR improvement under its own Q model.
- Stage 6 exact-v5 similarly disagrees: Extra Trees improves DR but reduces WIS;
  XGBoost improves DR and WIS but has worse factual and HIT-ranking metrics.
- Stage 5 has only two training Stages and one held-out Stage; its OPE is
  especially unstable. No Stage 5 candidate is ready for physical promotion.

Extra Trees exact-v5 is therefore the first *diagnostic experiment* for Stage
6 because it wins both factual MAE and HIT ranking in the contract-exact view.
The support-32/margin variants are safer experiment orders than unconstrained
argmax. This ranking is not a deployment decision: there is only one held-out
Stage, no Stage-level confidence interval, and physical play remains final
evidence.

## Collection recommendations

1. Collect transition-v5 complete Stages for Stages 1-3 first; this snapshot has
   none. Do not pool across stage, difficulty, character, or shot type.
2. Raise every active scope to at least ten clean v5 complete Stages, reserving
   at least three later Stages for untouched evaluation. Stage 4 and Stage 5
   currently have zero exact-v5 training rows; Stage 6 has only three v5 Stages.
3. Investigate Stage 5 collection before adding volume. Two of five runs are
   not training-eligible and the scope contains 1,876 dead-end rows. Its raw
   factual ratio is only 63.88%, compared with 99.19% on Lunatic Stage 4.
4. Add bounded native-safe exploration in low-support action/source-context
   cells, especially Stage 5. Record the exact propensity. Never enlarge the
   safe set, weaken fail-close behavior, or explore Bomb.
5. Keep v5 compact policy context on every transition and retain full physical
   frame roots. Sample frame shards by stage, source context, bullet-density
   bin, failure reason, and latency tail for recurring coherence checks.
6. Promote only after rolling chronological complete-Stage evaluation with
   per-Stage confidence intervals, followed by controlled physical A/B trials.
   Physical trials retain the first-HIT/authority/Bomb stop rules and the fresh
   issue revalidation boundary.

## Reproduce

Build only the Linux native library on a host without MinGW and install the CPU
dependencies:

```bash
cmake -S native -B build/native -DCMAKE_BUILD_TYPE=Release
cmake --build build/native -j2
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements-cpu-train.txt
```

Fetch compact streams at the fixed revision (the dataset is currently public,
so no token is required):

```bash
revision=211374189e2ec074dafe817d529fd3de833dafd4
local_dir=corpus/hf-$revision
for pattern in dataset_manifest.json README.md 'runs/*/run.json' \
  'runs/*/manifest.json' 'runs/*/infra-audit.json' \
  'runs/*/events-*.jsonl.gz' 'runs/*/transitions-*.jsonl.gz'; do
  nice -n 15 ionice -c2 -n7 hf download \
    Joh1rreq/touhou-solver-th06-rl-corpus --repo-type dataset \
    --revision "$revision" --local-dir "$local_dir" --max-workers 4 \
    --include "$pattern"
done
```

Run the full compact audit, native benchmark, and a scoped policy zoo:

```bash
nice -n 15 ionice -c2 -n7 env PYTHONPATH=src .venv/bin/python \
  scripts/audit_offline_corpus.py "$local_dir" --revision "$revision" \
  --output "artifacts/offline/corpus-audit-$revision.json"

nice -n 15 ionice -c2 -n7 env PYTHONPATH=src .venv/bin/python \
  scripts/benchmark_native_kernel.py --source reference/GensokyoClub-th06 \
  --output artifacts/offline/native-benchmark.json

nice -n 15 ionice -c2 -n7 env PYTHONPATH=src .venv/bin/python \
  scripts/train_offline_cpu.py "$local_dir" --revision "$revision" \
  --output artifacts/offline/stage6-exact-policy-zoo --scope 3/0/0/6 \
  --view exact-v5 --threads 12 --iterations 500 --max-train-rows 250000
```

`benchmark_recorded_frames.py` additionally requires selected frame shards from
one v5 run. `summarize_offline_benchmarks.py` then consolidates the corpus,
native, recorded-frame, and one or more policy manifests into a single ignored
JSON report. The consolidated report for this run is
`artifacts/offline/benchmark-suite-211374189e2e.json`.
