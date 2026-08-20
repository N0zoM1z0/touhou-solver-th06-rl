#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository="$(cd -- "${script_dir}/.." && pwd)"
python="${repository}/.venv/bin/python"

if [[ ! -x "${python}" ]]; then
    echo "missing ${python}; run scripts/bootstrap_wine_runtime.sh first" >&2
    exit 2
fi

artifact_dir="${TH06_RL_SMOKE_ARTIFACT_DIR:-${repository}/artifacts/wine-runtime-smoke-$(date -u +%Y%m%dT%H%M%SZ)-$$}"

set +e
"${python}" "${script_dir}/run_wine_retail.py" \
    "$@" \
    --practice-stage 1 \
    --difficulty lunatic \
    --seconds "${TH06_RL_SMOKE_SECONDS:-12}" \
    --policy-plugin "${script_dir}/policies/runtime_smoke_policy.py" \
    --policy-state "${repository}/config/runtime_smoke_policy.json" \
    --immutable-policy \
    --artifact-dir "${artifact_dir}"
runner_status=$?
set -e

if [[ ! -f "${artifact_dir}/report.json" ]]; then
    echo "Wine runner did not produce ${artifact_dir}/report.json" >&2
    if ((runner_status != 0)); then
        exit "${runner_status}"
    fi
    exit 1
fi
set +e
"${python}" "${script_dir}/verify_wine_smoke.py" "${artifact_dir}/report.json"
verification_status=$?
set -e
if ((runner_status != 0)); then
    exit "${runner_status}"
fi
exit "${verification_status}"
