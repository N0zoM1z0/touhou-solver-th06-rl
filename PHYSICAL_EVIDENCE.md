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

The mixed-revision row is retained because it is useful physical evidence,
but it is not a reproducible release qualification. New qualification runs
must start from a clean committed worktree and record that commit in `run.json`.
