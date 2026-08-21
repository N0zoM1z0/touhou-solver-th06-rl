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
