#!/bin/sh
set -eu

repository=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
build_dir="$repository/paper/build"

if [ -d "$build_dir" ]; then
    rm -rf -- "$build_dir"
fi
