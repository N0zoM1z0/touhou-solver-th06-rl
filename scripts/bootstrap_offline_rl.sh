#!/bin/sh
set -eu

repository=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
venv=${TH06_RL_VENV:-"$repository/.venv"}
python=${PYTHON:-python3}

if [ ! -x "$venv/bin/python" ]; then
    "$python" -m venv "$venv"
fi

"$venv/bin/python" -m pip install --upgrade pip
"$venv/bin/python" -m pip install -e "$repository[dev,offline]"

if [ "$(uname -s)" = Linux ] && [ "${TH06_RL_SKIP_TORCH:-0}" != 1 ]; then
    "$venv/bin/python" -m pip install \
        'torch==2.8.0+cpu' \
        --index-url https://download.pytorch.org/whl/cpu
fi

"$venv/bin/python" - <<'PY'
import importlib

required = ("numpy", "xgboost")
for name in required:
    module = importlib.import_module(name)
    print(f"{name}={module.__version__}")
try:
    torch = importlib.import_module("torch")
except ModuleNotFoundError:
    print("torch=skipped")
else:
    print(f"torch={torch.__version__}; cuda={torch.cuda.is_available()}")
PY
