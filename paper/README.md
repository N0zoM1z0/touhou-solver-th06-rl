# Research paper

The paper is maintained in LaTeX. From any directory, run:

```sh
./scripts/check_paper.sh
```

The canonical, tracked PDF is `paper/main.pdf`. Intermediate files stay under
the ignored `paper/build/` directory. `./scripts/clean_paper.sh` removes only
that generated directory and leaves the tracked PDF intact.

The document is a preregistered working draft: open experiments remain marked
open, and no result is promoted without an immutable Wine artifact.

`main.tex` is also the canonical algorithm decision record. It distinguishes
the NMNB goal metric from the initial expected-HIT optimization target, records
the causal decision-epoch learner unit, and gives every candidate representation,
algorithm, objective, and collection mechanism an explicit admission gate.
Speculative candidates are not active implementation merely because they are
listed in the paper.
