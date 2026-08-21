#!/bin/sh
set -eu

repository=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
paper_dir="$repository/paper"

if ! command -v latexmk >/dev/null 2>&1; then
    echo "latexmk is required (Debian/Ubuntu: apt install latexmk texlive-latex-extra)" >&2
    exit 1
fi

mkdir -p "$paper_dir/build"
cd "$paper_dir"
latexmk \
    -pdf \
    -silent \
    -interaction=nonstopmode \
    -halt-on-error \
    -file-line-error \
    -outdir=build \
    main.tex

staged_pdf=$(mktemp "$paper_dir/.main.pdf.XXXXXX")
trap 'rm -f -- "$staged_pdf"' EXIT
cp -- "$paper_dir/build/main.pdf" "$staged_pdf"
mv -- "$staged_pdf" "$paper_dir/main.pdf"
trap - EXIT
