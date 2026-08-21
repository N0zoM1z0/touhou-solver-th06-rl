#!/bin/sh
set -eu

repository=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
"$repository/scripts/build_paper.sh"

log="$repository/paper/build/main.log"
pdf="$repository/paper/main.pdf"
test -s "$pdf"

if grep -E \
    'LaTeX Warning: (Citation|Reference).*undefined|There were undefined references|multiply defined' \
    "$log"; then
    echo "paper contains unresolved or multiply defined references" >&2
    exit 1
fi

echo "paper check passed: $pdf"
