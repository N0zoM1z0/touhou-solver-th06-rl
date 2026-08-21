# Experiment work logs

Runtime experiment narration lives under the ignored `work_log/` directory.
Each experiment creates one directory named with its first UTC start time and
experiment identifier, for example:

```text
work_log/20260821T130000Z-l1-stage4-bc-v1/
  session.json
  events.jsonl
```

`session.json` binds the log to the repository commit, preregistration path and
hash, and immutable schedule. `events.jsonl` appends coarse experiment events:
episode attempts, admission or rejection, fit, online canary, and the final
decision. Per-process stdout, Wine traces, corpora, audits, fitted models, and
result ledgers remain in their declared ignored artifact locations.

Work logs are navigation and operational context, not scientific authority.
Claims must cite the immutable corpus, audit, model, and result hashes. The
directory is intentionally ignored because machine logs and generated evidence
must not be committed.
