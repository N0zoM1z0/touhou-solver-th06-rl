#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository="$(cd -- "${script_dir}/.." && pwd)"
archive="${repository}/../game-exe/th06.rar"
game_dir="${repository}/reference/th06-game-original/th06"
wine_prefix="${repository}/reference/wine-prefixes/th06-retail"
exact_source=""
install_system_packages=1

retail_sha256="9f76483c46256804792399296619c1274363c31cd8f1775fafb55106fb852245"
score_sha256="54cd436d5d8a7a904190c792a977bf270ab1cb759fd72101e51e94d26b749c71"
python_zip_sha256="daf24de7fb3b173e94e56a201d3f38dfedebbdc7ed1925f7aeb8ed588e2b4189"
python_exe_sha256="e60592888c3128132df3489a2462716bb268063bfe3564bfe1f2f3dbe9ceafd1"
exact_source_commit="cc475a0bc3fef38683b0f02224c87ddba0a021d9"
python_url="https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-win32.zip"

usage() {
    printf '%s\n' \
        'Bootstrap the self-contained original-retail Wine runtime.' \
        '' \
        'Usage: scripts/bootstrap_wine_runtime.sh [options]' \
        '' \
        'Options:' \
        '  --archive PATH         TH06 archive (default: sibling game-exe/th06.rar)' \
        '  --game-dir PATH        ignored extracted game directory' \
        '  --wine-prefix PATH     ignored dedicated 32-bit Wine prefix' \
        '  --exact-source PATH    optionally attest a GensokyoClub/th06 checkout' \
        '  --skip-system-packages do not invoke apt/dpkg' \
        '  -h, --help             show this help'
}

# Bootstrap the self-contained original-retail Wine runtime.
#
# Usage: scripts/bootstrap_wine_runtime.sh [options]
#
# Options:
#   --archive PATH         TH06 archive (default: sibling game-exe/th06.rar)
#   --game-dir PATH        ignored extracted game directory
#   --wine-prefix PATH     ignored dedicated 32-bit Wine prefix
#   --exact-source PATH    optionally attest a GensokyoClub/th06 checkout
#   --skip-system-packages do not invoke apt/dpkg
#   -h, --help             show this help
while (($#)); do
    case "$1" in
        --archive)
            archive="$2"
            shift 2
            ;;
        --game-dir)
            game_dir="$2"
            shift 2
            ;;
        --wine-prefix)
            wine_prefix="$2"
            shift 2
            ;;
        --exact-source)
            exact_source="$2"
            shift 2
            ;;
        --skip-system-packages)
            install_system_packages=0
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "unknown option: $1" >&2
            exit 2
            ;;
    esac
done

if ((install_system_packages)); then
    if ! command -v apt-get >/dev/null || ! command -v dpkg >/dev/null; then
        echo "automatic package installation currently supports Debian/Ubuntu only" >&2
        exit 2
    fi
    root_command=()
    if ((EUID != 0)); then
        if ! command -v sudo >/dev/null; then
            echo "sudo is required for system package installation" >&2
            exit 2
        fi
        root_command=(sudo)
    fi
    "${root_command[@]}" dpkg --add-architecture i386
    "${root_command[@]}" apt-get update
    "${root_command[@]}" env DEBIAN_FRONTEND=noninteractive apt-get install -y \
        ca-certificates cmake curl g++ g++-mingw-w64-i686 gdb git \
        libasound2:i386 libasound2-plugins:i386 libgl1:i386 \
        libgl1-mesa-dri:i386 locales-all mesa-utils ninja-build \
        python3 python3-pip python3-venv sudo unar unzip util-linux wine \
        wine32:i386 wine64 xauth xdotool xvfb
fi

required_commands=(cmake curl gdb git i686-w64-mingw32-g++ ninja python3 unar unzip wine wineboot wineserver Xvfb)
if ((EUID != 0)); then
    required_commands+=(sudo)
fi
for command_name in "${required_commands[@]}"; do
    if ! command -v "${command_name}" >/dev/null; then
        echo "required command is absent: ${command_name}" >&2
        exit 2
    fi
done
if ((EUID != 0)) && ! sudo -n true >/dev/null 2>&1; then
    echo "unattended GDB normalization requires passwordless sudo (sudo -n failed)" >&2
    exit 2
fi
if [[ ! -f "${archive}" ]]; then
    echo "TH06 archive is absent: ${archive}" >&2
    exit 2
fi

if [[ -n "${exact_source}" ]]; then
    if ! git -C "${exact_source}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        echo "exact-source checkout is not a Git repository: ${exact_source}" >&2
        exit 2
    fi
    observed_source_commit="$(git -C "${exact_source}" rev-parse HEAD)"
    if [[ "${observed_source_commit}" != "${exact_source_commit}" ]]; then
        echo "exact-source commit mismatch: ${observed_source_commit}" >&2
        exit 2
    fi
    echo "attested exact-source commit ${observed_source_commit}"
fi

if [[ -f "${game_dir}/東方紅魔郷.exe" ]]; then
    observed_retail_sha256="$(sha256sum "${game_dir}/東方紅魔郷.exe" | awk '{print $1}')"
    if [[ "${observed_retail_sha256}" != "${retail_sha256}" ]]; then
        echo "existing retail executable SHA-256 mismatch: ${observed_retail_sha256}" >&2
        exit 2
    fi
    echo "retail game already attested at ${game_dir}"
else
    if [[ -e "${game_dir}" ]]; then
        echo "game directory exists without the attested executable: ${game_dir}" >&2
        exit 2
    fi
    mkdir -p -- "$(dirname -- "${game_dir}")"
    extraction_root="$(mktemp -d "$(dirname -- "${game_dir}")/.th06-extract.XXXXXX")"
    trap 'rm -rf -- "${extraction_root}"' EXIT
    unar -quiet -force-overwrite -output-directory "${extraction_root}" "${archive}"
    mapfile -d '' retail_candidates < <(
        find "${extraction_root}" -type f -name '東方紅魔郷.exe' -print0
    )
    if ((${#retail_candidates[@]} != 1)); then
        echo "expected one canonical retail executable in archive, found ${#retail_candidates[@]}" >&2
        exit 2
    fi
    observed_retail_sha256="$(sha256sum "${retail_candidates[0]}" | awk '{print $1}')"
    if [[ "${observed_retail_sha256}" != "${retail_sha256}" ]]; then
        echo "archive retail executable SHA-256 mismatch: ${observed_retail_sha256}" >&2
        exit 2
    fi
    mv -- "$(dirname -- "${retail_candidates[0]}")" "${game_dir}"
    rm -rf -- "${extraction_root}"
    trap - EXIT
    echo "extracted and attested retail game at ${game_dir}"
fi

score_template="$(dirname -- "${game_dir}")/full-unlock-score.dat"
if [[ -f "${score_template}" ]]; then
    observed_score_sha256="$(sha256sum "${score_template}" | awk '{print $1}')"
    if [[ "${observed_score_sha256}" != "${score_sha256}" ]]; then
        echo "existing score template SHA-256 mismatch: ${observed_score_sha256}" >&2
        exit 2
    fi
else
    mapfile -d '' score_candidates < <(
        find "${game_dir}" -type f -path '*/全开档/score.dat' -print0
    )
    if ((${#score_candidates[@]} != 1)); then
        echo "expected one full-unlock score.dat, found ${#score_candidates[@]}" >&2
        exit 2
    fi
    observed_score_sha256="$(sha256sum "${score_candidates[0]}" | awk '{print $1}')"
    if [[ "${observed_score_sha256}" != "${score_sha256}" ]]; then
        echo "full-unlock score SHA-256 mismatch: ${observed_score_sha256}" >&2
        exit 2
    fi
    cp -- "${score_candidates[0]}" "${score_template}"
fi

if [[ ! -x "${repository}/.venv/bin/python" ]]; then
    python3 -m venv "${repository}/.venv"
fi
"${repository}/.venv/bin/python" -m pip install --upgrade pip setuptools wheel
"${repository}/.venv/bin/python" -m pip install -e "${repository}[dev]"

tools_dir="${repository}/reference/tools"
python_zip="${tools_dir}/python-3.11.9-embed-win32.zip"
python_runtime="${tools_dir}/windows-python-3.11.9-embed-win32"
mkdir -p -- "${tools_dir}"
if [[ ! -f "${python_zip}" ]]; then
    python_download="$(mktemp "${tools_dir}/.python-embed.XXXXXX.zip")"
    trap 'rm -f -- "${python_download}"' EXIT
    curl --fail --location --retry 3 --output "${python_download}" "${python_url}"
    observed_python_zip_sha256="$(sha256sum "${python_download}" | awk '{print $1}')"
    if [[ "${observed_python_zip_sha256}" != "${python_zip_sha256}" ]]; then
        echo "Windows Python archive SHA-256 mismatch: ${observed_python_zip_sha256}" >&2
        exit 2
    fi
    mv -- "${python_download}" "${python_zip}"
    trap - EXIT
fi
observed_python_zip_sha256="$(sha256sum "${python_zip}" | awk '{print $1}')"
if [[ "${observed_python_zip_sha256}" != "${python_zip_sha256}" ]]; then
    echo "existing Windows Python archive SHA-256 mismatch: ${observed_python_zip_sha256}" >&2
    exit 2
fi
if [[ ! -d "${python_runtime}" ]]; then
    python_extraction="$(mktemp -d "${tools_dir}/.python-embed.XXXXXX")"
    trap 'rm -rf -- "${python_extraction}"' EXIT
    unzip -q "${python_zip}" -d "${python_extraction}"
    mv -- "${python_extraction}" "${python_runtime}"
    trap - EXIT
fi
if [[ ! -f "${python_runtime}/python.exe" ]]; then
    echo "Windows embeddable Python is incomplete: ${python_runtime}" >&2
    exit 2
fi
observed_python_exe_sha256="$(sha256sum "${python_runtime}/python.exe" | awk '{print $1}')"
if [[ "${observed_python_exe_sha256}" != "${python_exe_sha256}" ]]; then
    echo "Windows Python executable SHA-256 mismatch: ${observed_python_exe_sha256}" >&2
    exit 2
fi
"${repository}/.venv/bin/python" "${script_dir}/configure_embedded_python.py" \
    "${python_runtime}" --repository "${repository}"

"${repository}/.venv/bin/python" "${script_dir}/configure_wine_retail.py" \
    "${game_dir}" --initialize

cmake -S "${repository}/native" -B "${repository}/build/native" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "${repository}/build/native" --parallel
cmake -S "${repository}/native" \
    -B "${repository}/build/native-win32-fully-static" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE="${repository}/cmake/toolchains/mingw32.cmake"
cmake --build "${repository}/build/native-win32-fully-static" --parallel

native_dll="${repository}/build/native-win32-fully-static/libth06_rl_native.dll"
if [[ ! -f "${native_dll}" ]]; then
    echo "Win32 native library was not built: ${native_dll}" >&2
    exit 2
fi

mkdir -p -- "$(dirname -- "${wine_prefix}")"
echo
echo "portable Wine runtime provisioned"
echo "  game:         ${game_dir}"
echo "  Wine prefix:  ${wine_prefix}"
echo "  Win32 Python: ${python_runtime}/python.exe"
echo "  native DLL:   ${native_dll}"
echo
echo "Run the end-to-end game/controller smoke with:"
echo "  ${script_dir}/smoke_wine_runtime.sh"
