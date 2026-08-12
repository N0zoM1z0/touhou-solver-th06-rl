# Offline RL reference cache

The local `reference/offline-rl/` tree is an ignored, reproducible research
cache. Papers and upstream repositories are deliberately not committed: the
tracked manifest `config/offline_rl_references.json` binds every paper to its
publisher URL and SHA-256 and every repository to its origin URL, exact commit,
and observed license status.

Rebuild the cache from a clean checkout with:

```bash
python3 scripts/fetch_offline_rl_references.py
```

Verify an existing cache without network writes with:

```bash
python3 scripts/fetch_offline_rl_references.py --verify-only
```

The downloader refuses to overwrite a mismatched file or silently move a Git
checkout. A changed upstream artifact therefore requires an explicit manifest
review and a new commit, rather than silently changing the evidence behind an
algorithm decision.

## Bound sources and use

- Implicit Q-Learning, arXiv 2110.06169v1, and the authors' MIT repository:
  reference for in-sample Bellman learning and expectile value fitting without
  querying unseen actions.
- Conservative Q-Learning, NeurIPS 2020, and the authors' repository: reference
  for pessimism under offline distribution shift. The checked-out repository
  has no license file, so it is read-only research material and no code may be
  copied from it.
- Safe Policy Improvement with Baseline Bootstrapping, ICML 2019, and its MIT
  repository: reference for falling back to an incumbent where data support is
  insufficient.
- Batch Policy Learning under Constraints, ICML 2019, and its authors'
  repository: reference for fitted policy evaluation. The checked-out
  repository has no license file and is read-only research material.
- Doubly Robust Off-policy Value Evaluation for Reinforcement Learning, ICML
  2016: reference for sequential off-policy evaluation and its variance
  trade-offs, not a license to reuse complete-Stage importance-weighted labels.

These papers inform generic algorithm design. They do not change the product
contract: factual original-Wine transitions only, physical HIT-only cost,
native-safe actions, episode-grouped validation, and full Wine canary/final
evaluation. Any formula implemented here is independently written and covered
by repository tests.
