# Self-contained retail adapter provenance

The runtime adapter under `src/th06_rl/retail/` is owned and packaged by this
repository. It has no import, filesystem, or installation dependency on the
historical `N0zoM1z0/touhou-solver-th06` repository.

The initial adapter implementation was migrated on 2026-08-20 from commit
`6ff22d9552dac101246cc33fd58bc4af54e40ff8` of that same-owner historical
repository. The migration retained only the low-level TH06 1.02h retail
process, input, data-model, and source-semantics modules needed by the current
Wine runtime. Scripted routes and gameplay policy code were not migrated.

The adapter is bound to these independently checked identities:

- exact-source checkout commit:
  `cc475a0bc3fef38683b0f02224c87ddba0a021d9`;
- original Japanese executable SHA-256:
  `9f76483c46256804792399296619c1274363c31cd8f1775fafb55106fb852245`.

`tests/test_repository_prune.py` enforces the package boundary: active Python
must not import a top-level `th06` package, and the former path-injection helper
must remain absent. Source checkouts and original-game assets stay ignored and
read-only; setup materializes them only inside ignored runtime directories.
