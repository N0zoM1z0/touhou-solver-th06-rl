# Hard-empty source audit

## Verdict

The native collision geometry agrees with the shipped TH06 collision shape and
update order. The controller's old empty-set interpretation did not: it
treated the repo's additional 0.35 px uncertainty margin as if it were part of
the game's physical collision contract.

The lossless generation-1 Wine roots give direct evidence. Of 10 recorded
`Hard safe set empty` roots:

- 7 remained empty under source-exact geometry;
- 3 had a nonempty source-exact set after the 0.35 px conservative set closed;
- those 3 roots retained 9, 2, and 12 source-legal actions respectively;
- recorded and replayed 0.35 px masks agreed at all 10 roots.

Therefore native hazard geometry is not changed. The controller keeps the
0.35 px set as its normal preference and, only when that set is empty, asks the
same native kernel for the 0 px source-exact set. It reports a true Hard-empty
only if both sets are empty. Lookahead, profiling, and final fresh
certification use the selected contract consistently for that decision.

This is infrastructure repair, not learned-policy shaping: no pattern, phase,
RNG, reward, action, or model threshold participates in the fallback.

## Authoritative source binding

The audit is bound to GensokyoClub/th06 commit
`cc475a0bc3fef38683b0f02224c87ddba0a021d9` and records SHA-256 values for all
source files it relies on.

The relevant source facts are:

1. `Player.cpp` initializes the player half-hitbox to `(1.25, 1.25)`.
2. `Player::HandlePlayerInputs` moves and clamps the player, then updates the
   hitbox corners.
3. `ChainPriorities.hpp` places Player update at priority 7 and BulletManager
   at priority 11, so player movement precedes bullet motion/collision in a
   game frame.
4. `BulletManager.cpp` advances a fired bullet before calling
   `Player::CalcKillBoxCollision`.
5. `Player::CalcKillBoxCollision` uses inclusive AABB overlap and applies no
   extra collision margin.
6. Laser and collidable-enemy contact ultimately use the same player hitbox;
   the repo retains the shipped laser midpoint-hitbox behavior and the enemy
   hitbox scaling visible in source.

The reproducible command is:

```bash
.venv/bin/python scripts/audit_wine_hard_empty.py \
  artifacts/autonomous-wine-generation-1/collection-corpus/* \
  --native-library build/native/libth06_rl_native.so \
  --output artifacts/autonomous-wine-generation-1/hard-empty-source-audit-v2.json
```

The ignored JSON artifact contains the source commit, source-file hashes,
per-action clearances at horizons 1 through 4, conservative/source collision
witnesses, exact action sets, and any factual continuous-stage follow-up.

## Complete-stage follow-up

Generation 1 stopped at Hard-empty, so it could not provide the physical
follow-up needed to distinguish eventual HIT from recovery. Generation 2
completed eight full Wine stages with continue-on-HIT and supplied that
evidence. The source-bound audit examined 1,468 recorded true Hard-empty
decision roots:

- recorded and recomputed conservative masks agreed at all 1,468 roots;
- all 1,468 remained empty under source-exact margin-0 geometry;
- zero were conservative-margin-only closures;
- 809 were followed by a physical HIT in 1 to 9 frames, mean 2.42;
- 659 recovered a native safe set in 1 to 76 frames, mean 6.20;
- all eight complete-stage run audits passed.

Consecutive control frames may each contribute a root, so these are not 1,468
independent episodes. They are continuous factual follow-ups, and they show
that the fallback repaired the margin-only false-empty class without hiding
true source-geometry closures. There is no evidence-backed Hard-empty
infrastructure change to make from this evidence. Historical learner verdicts
are consolidated in `LEARNER_AUDIT_AND_GENERATION7_DECISION.md`; the ignored
audit artifacts remain hash-bound local evidence.

An already-issued input lease that becomes source-unsafe on the next coherent
Wine frame is likewise a control dead-end, not an infrastructure failure. The
controller releases the in-flight input and records it separately as
`control-dead-end:in-flight input unsafe`; it never converts that event into a
HIT cost or silently treats the desired action as safe.
