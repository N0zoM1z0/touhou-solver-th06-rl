#!/usr/bin/env bash
set -euo pipefail

TH06_RL_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TH06_RL_BUILD="$TH06_RL_REPO/build"
mkdir -p "$TH06_RL_BUILD"

cmake -S "$TH06_RL_REPO/native" -B "$TH06_RL_BUILD/native" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$TH06_RL_BUILD/native" -j2

x86_64-w64-mingw32-g++ \
  -std=c++20 -O3 -Wall -Wextra -Werror \
  -shared -static -static-libgcc -static-libstdc++ \
  -DTH06_RL_NATIVE_BUILD=1 \
  -I"$TH06_RL_REPO/native/include" \
  "$TH06_RL_REPO/native/src/th06_rl_native.cpp" \
  -o "$TH06_RL_BUILD/th06_rl_native.dll"

sha256sum \
  "$TH06_RL_BUILD/native/libth06_rl_native.so" \
  "$TH06_RL_BUILD/th06_rl_native.dll"

