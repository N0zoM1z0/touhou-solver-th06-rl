# Physical evidence ledger

Only complete physical Practice stages belong here. Corpus, traces, game
files, and generated audits remain ignored; the run ID is the lookup key for
local or mirrored evidence.

Physical validation commits use these trailers when applicable:

```text
Physical-Route: <difficulty>/<character-shot>/Stage<stage>
Physical-Run: <run-id>
Physical-Hits: <count>
Capture-P95-ms: <value>
Stale-Rate: <fraction>
Dense-Control-P95-ms: <value>
Infra-Audit: <compact classification summary>
```

| Date | Route | Run | Runtime revision | HIT | Capture P95 | Stale | Infra audit |
|---|---|---|---|---:|---:|---:|---|
| 2026-08-06 | Hard / Reimu-A / Stage 4 | `20260806T102838Z-125511400` | mixed transition run: run metadata `fb0aaef`; lightweight controller later committed as `b6ebb21`; policy hot-reloaded through `1661600` | 3 | 8.643 ms | 2.395% | complete; 3/3 HIT preceded by empty Hard set; 0 geometry/certificate candidates; 0 Bomb; 28 authoritative anchors |
| 2026-08-06 | Lunatic / Reimu-A / Stage 4 | `20260806T104008Z-119102900` | `50929d8` | 4 | 6.483 ms | 0.0159% | complete; 4/4 HIT preceded by empty Hard set; dense (500+ bullets) control P95 9.779 ms; full-decode Hard parity 64/64; 0 geometry/certificate candidates; 0 Bomb; 26 authoritative anchors; 233 MiB compressed corpus |
| 2026-08-06 | Lunatic / Reimu-A / Stage 4 | `20260806T105403Z-514814600` | `5f30acb` | 3 | 6.442 ms | 0.0225% | complete; 1 empty Hard set; 1 newly spawned overlapping boss body; 1 checkpoint-correlated 8-frame latency gap; full-decode Hard parity 64/64; 0 geometry/certificate candidates; 0 Bomb; 25 authoritative anchors; 249 MiB compressed corpus |
| 2026-08-06 | Lunatic / Reimu-A / Stage 4 | `20260806T112143Z-881624900` | controller `1da4ff2`; checkpoint-only policy codec hot-reloaded to `e132ab2` | 1 | 7.186 ms | 0.0745% | complete; sole HIT preceded by empty Hard set; full-decode Hard parity 64/64; 0 geometry/certificate candidates; 0 Bomb; 245 MiB compressed corpus |
| 2026-08-06 | Lunatic / Reimu-A / Stage 4 | `20260806T114043Z-165645200` | `756fa7b` | 4 | 6.245 ms | 0% | complete; 4/4 HIT preceded by empty Hard set; full-decode Hard parity 64/64; 0 geometry/certificate candidates; 0 Bomb; 25 authoritative anchors; 234 MiB compressed corpus |
| 2026-08-06 | Lunatic / Reimu-A / Stage 5 | `20260806T114830Z-235638200` | `756fa7b` | 5 | 6.004 ms | 0% | complete cold-start episode; 5/5 HIT preceded by empty Hard set; full-decode Hard parity 64/64; 0 geometry/certificate candidates; 0 Bomb; 28 authoritative anchors; 118 MiB compressed corpus |
| 2026-08-06 | Lunatic / Reimu-A / Stage 6 | `20260806T115616Z-821716900` | `756fa7b` | 16 | 6.078 ms | 0.958% | complete cold-start episode; 16/16 HIT preceded by empty Hard set; full-decode Hard parity 62 checked frames with 0 divergence; 0 geometry/certificate candidates; 0 Bomb; 78 authoritative anchors; 309 MiB compressed corpus |
| 2026-08-06 | Lunatic / Reimu-A / Stage 4 | `20260806T120632Z-280200100` | `756fa7b` | 3 | 5.811 ms | 0.0075% | complete; 3/3 HIT preceded by empty Hard set; full-decode Hard parity 64/64; 0 geometry/certificate candidates; 0 Bomb; 25 authoritative anchors; 244 MiB compressed corpus |
| 2026-08-06 | Lunatic / Reimu-A / Stage 6 | `20260806T145106Z-285711500` | `eb290c7` | 21 | 7.048 ms | 0.1061% | complete first hierarchical-v2 feedback episode; 21/21 HIT preceded by empty Hard set; dense (500+ bullets) control P95 9.702 ms with 0 stale retries; full-decode Hard parity 64/64; 0 geometry/certificate candidates; 0 Bomb; 89 authoritative anchors; 289 MiB compressed corpus |

The mixed-revision row is retained because it is useful physical evidence,
but it is not a reproducible release qualification. New qualification runs
must start from a clean committed worktree and record that commit in `run.json`.
