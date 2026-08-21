# TH06 ECL reference cache

The seven retail stage ECL files can be rebuilt from an owned TH06 1.02h
`紅魔郷ST.DAT` without a source checkout or the historical solver:

```bash
python3 scripts/extract_th06_ecl.py \
  --archive /path/to/th06/紅魔郷ST.DAT
```

The default output is the ignored `reference/th06-ecl-original/` directory.
The tracked `config/th06_ecl_reference.json` pins the input archive and every
decoded payload by SHA-256 and size. To check an existing cache without
writing it:

```bash
python3 scripts/extract_th06_ecl.py \
  --archive /path/to/th06/紅魔郷ST.DAT \
  --verify-only
```

These payloads are source/reference evidence only. They are not corpus rows,
training examples, runtime dependencies, or tracked game assets.
