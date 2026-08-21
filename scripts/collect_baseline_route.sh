#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository="$(cd -- "${script_dir}/.." && pwd)"
python="${repository}/.venv/bin/python"
label="${TH06_RL_BASELINE_LABEL:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
artifact_dir="${TH06_RL_BASELINE_ARTIFACT_DIR:-${repository}/artifacts/baseline-route-${label}}"
corpus_root="${TH06_RL_BASELINE_CORPUS_ROOT:-${repository}/corpora/baseline-route-${label}}"

if [[ ! -x "${python}" ]]; then
    echo "missing ${python}; run scripts/bootstrap_wine_runtime.sh first" >&2
    exit 2
fi

set +e
"${python}" "${script_dir}/run_wine_retail.py" \
    "$@" \
    --start-route \
    --difficulty lunatic \
    --complete-route-corpus-root "${corpus_root}" \
    --policy-plugin "${script_dir}/policies/runtime_smoke_policy.py" \
    --policy-state "${repository}/config/runtime_smoke_policy.json" \
    --immutable-policy \
    --artifact-dir "${artifact_dir}"
runner_status=$?
set -e

mapfile -d '' manifests < <(find "${corpus_root}" -mindepth 2 -maxdepth 2 -name manifest.json -print0 2>/dev/null)
if ((${#manifests[@]} != 1)); then
    echo "expected one route corpus manifest, found ${#manifests[@]}" >&2
    if ((runner_status != 0)); then
        exit "${runner_status}"
    fi
    exit 1
fi
run_dir="$(dirname -- "${manifests[0]}")"
audit_path="${artifact_dir}/infra-audit.json"
set +e
"${python}" "${script_dir}/audit_run.py" "${run_dir}" \
    --native-library "${repository}/build/native/libth06_rl_native.so" \
    --output "${audit_path}"
audit_status=$?
set -e

if ((runner_status != 0)); then
    exit "${runner_status}"
fi
if ((audit_status != 0)); then
    exit "${audit_status}"
fi
"${python}" "${script_dir}/verify_baseline_route.py" \
    "${artifact_dir}/report.json" \
    "${run_dir}/run.json" \
    "${run_dir}/manifest.json" \
    "${audit_path}"
