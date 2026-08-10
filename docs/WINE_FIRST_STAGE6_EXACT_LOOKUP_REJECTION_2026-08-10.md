# Stage 6 exact lookup feasibility rejection (2026-08-10)

## Result

Do not build a residual that maps the three positive sub10 Wine anchors by
their exact frozen-UCB fine context keys.  Replaying those three keys over the
current-kernel r1-r8 factual prefixes produced only 3 activations in 36,106
policy calls (`0.0000830887`).  Each key occurred only at the same physical
episode and anchor from which its COW winner was learned; none recurred in an
independent episode.

The three memorized mappings would have been:

- r1 `hard:33bfd|legal:33bfd` -> `up_right`;
- r2 `hard:35faf|legal:35faf` -> `down`;
- r8 `hard:15eaf|legal:15eaf` -> `up_left_fast`.

This is zero demonstrated cross-episode reuse.  It would also make the
resident decision depend on an accidental frontier identity rather than a
generic action-relative property.  The exact-key route therefore creates
zero residual candidates and cannot enter Wine shadow.

## Evidence boundary

The replay used the same 36,106-call factual r1-r8 population already sealed
by `framev5-frozen-ucb-r1-r8-factual-action-audit.json` (SHA-256
`c61cc4b07e657d5f143646baa507ead546401ed9c70c1e1bcc235c3955fbd646`).
This feasibility count is a reject-only diagnostic.  It is not training or
promotion evidence.  Complete natural original-retail Wine Stage 6 HIT count
remains the eventual acceptance metric.

