# Stage 6 Wine-first panel 2 protocol (predeclared 2026-08-10)

## Purpose

Collect exactly three new original-retail Wine first-failure episodes before
any fit, checkpoint selection, or policy change.  Panel 2 may discover a new
repeated generic failure opportunity.  It may not reopen either action pair
rejected by the closed frame-v5 panel:

- `down_right` versus `down_fast`;
- `up_fast` versus `down_fast`.

Those pairs remain rejected even if a later episode contains a favorable row.

## Frozen collection contract

All three runs use:

- Lunatic / Reimu-A / Stage 6 Practice;
- original retail 1.02h executable SHA-256
  `9f76483c46256804792399296619c1274363c31cd8f1775fafb55106fb852245`;
- native kernel SHA-256
  `71b27bef942928dfc04b6a489ef261cb6485cd5bab2cd4d4ebf4081991a22b5b`;
- `phase-local-hierarchical-ucb-v4`, immutable state, exploration `0`;
- frame schema v5 and lossless first-failure corpus collection;
- natural Practice start, no life patch, no HIT continuation;
- stop on first physical HIT, authority failure, or Bomb request;
- exactly three episodes, provisionally named r4, r5, and r6.

No training, residual construction, shadow state, active canary, threshold
change, or checkpoint selection occurs between the three episodes.

## Post-collection audit

After all three runs complete:

1. verify immutable policy/config/score hashes and exact process cleanup;
2. replay the frozen incumbent and require zero recorded-action mismatch;
3. group by physical episode, not frame count;
4. exclude both previously rejected action pairs;
5. require at least two panel-2 episodes for any new generic opportunity;
6. replay exact sampled dialogue inputs and keep RNG, draw/platform, geometry,
   delivery, and native-hard-set divergence as separate fields;
7. compare reconstructed native hard sets with the retail delivery envelope
   `(0, 1, 2, 3)`, while source STEP branches retain `(0,)`;
8. run at most one declared action pair per newly repeated region.

Headless COW remains reject-only.  Panel 2 alone cannot promote a residual;
any unanimous new hypothesis must still enter a disjoint original-retail Wine
shadow episode before active publication is considered.
