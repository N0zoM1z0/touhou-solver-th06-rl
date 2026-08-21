#!/usr/bin/env bash
set -euo pipefail

TH06_RL_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TH06_RL_BUILD="$TH06_RL_REPO/build"
mkdir -p "$TH06_RL_BUILD"

required=(cmake i686-w64-mingw32-g++)
for command_name in "${required[@]}"; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "required native build command is absent: $command_name" >&2
    exit 2
  fi
done

cmake -S "$TH06_RL_REPO/native" -B "$TH06_RL_BUILD/native" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$TH06_RL_BUILD/native" -j2

cmake -S "$TH06_RL_REPO/native" \
  -B "$TH06_RL_BUILD/native-win32-fully-static" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="$TH06_RL_REPO/cmake/toolchains/mingw32.cmake"
cmake --build "$TH06_RL_BUILD/native-win32-fully-static" -j2

sha256sum \
  "$TH06_RL_BUILD/native/libth06_rl_native.so" \
  "$TH06_RL_BUILD/native-win32-fully-static/libth06_rl_native.dll"
