#!/usr/bin/env python3
"""Compare the fully-static Wine actor kernel with its portable artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY / "src"))

from th06_rl.iql_actor_learning import (  # noqa: E402
    NATIVE_ACTOR_ABSOLUTE_TOLERANCE,
    NATIVE_ACTOR_FLOAT32_RELATIVE_TOLERANCE,
    iql_actor_model_from_artifact,
    native_actor_prediction_tolerance_ratio,
)


ARRAY_NAMES = (
    "state_hidden_weight", "state_hidden_bias",
    "state_latent_weight", "state_latent_bias",
    "action_hidden_weight", "action_hidden_bias",
    "action_latent_weight", "action_latent_bias",
    "action_score_weight", "action_score_bias",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--windows-library", required=True, type=Path)
    parser.add_argument(
        "--windows-python", type=Path,
        default=(
            REPOSITORY
            / "reference/tools/windows-python-3.11.9-embed-win32/python.exe"
        ),
    )
    parser.add_argument(
        "--output", required=True, type=Path
    )
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to replace smoke report: {args.output}")
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    models = [iql_actor_model_from_artifact(row) for row in candidate["actors"]]
    reference = models[0]
    state = np.asarray([
        math.sin(index * 0.173) for index in range(len(reference.state_mean))
    ], dtype=np.float32)
    actions = np.asarray([
        [math.cos((row + 1) * (column + 3) * 0.071)
         for column in range(len(reference.action_mean))]
        for row in range(18)
    ], dtype=np.float32)
    rows = np.zeros((len(actions), len(reference.layout.names)), dtype=np.float32)
    rows[:, reference.layout.state_indices] = (
        state * reference.state_scale + reference.state_mean
    )
    rows[:, reference.layout.action_indices] = (
        actions * reference.action_scale + reference.action_mean
    )
    expected = np.asarray([
        model.predict(rows) for model in models
    ], dtype=np.float32).reshape(-1)

    def flatten(name: str) -> list[float]:
        return np.concatenate([
            np.asarray(getattr(model, name), dtype=np.float32).reshape(-1)
            for model in models
        ]).tolist()

    fixture = {
        "state": state.tolist(),
        "actions": actions.reshape(-1).tolist(),
        "row_count": len(actions),
        "state_count": len(state),
        "action_count": actions.shape[1],
        "model_count": len(models),
        "hidden_count": len(reference.state_hidden_bias),
        "rank_count": len(reference.state_latent_bias),
        "array_names": list(ARRAY_NAMES),
        **{name: flatten(name) for name in ARRAY_NAMES},
    }
    with tempfile.TemporaryDirectory(prefix="th06-g6-win-native-") as temporary:
        fixture_path = Path(temporary) / "fixture.json"
        fixture_path.write_text(json.dumps(fixture), encoding="utf-8")
        def windows_path(path: Path) -> str:
            return subprocess.check_output(
                ["winepath", "-w", str(path.resolve())], text=True
            ).strip()
        command = [
            "wine", str(args.windows_python),
            windows_path(REPOSITORY / "scripts/windows_iql_actor_kernel.py"),
            "--fixture", windows_path(fixture_path),
            "--library", windows_path(args.windows_library),
        ]
        completed = subprocess.run(
            command, check=True, text=True, capture_output=True,
            env={**__import__("os").environ, "WINEDEBUG": "-all"},
        )
    output = json.loads(completed.stdout.strip().splitlines()[-1])
    actual = np.asarray(output["outputs"], dtype=np.float32)
    maximum_error = float(np.max(np.abs(actual - expected)))
    tolerance_ratio = native_actor_prediction_tolerance_ratio(expected, actual)
    report = {
        "schema": "autonomous-generation-6-windows-native-smoke-v1",
        "evidence_eligible": False,
        "candidate_sha256": hashlib.sha256(
            args.candidate.read_bytes()
        ).hexdigest(),
        "windows_library_sha256": hashlib.sha256(
            args.windows_library.read_bytes()
        ).hexdigest(),
        "models": len(models),
        "rows": len(actions),
        "outputs": len(actual),
        "maximum_prediction_error": maximum_error,
        "absolute_tolerance": NATIVE_ACTOR_ABSOLUTE_TOLERANCE,
        "float32_relative_tolerance": (
            NATIVE_ACTOR_FLOAT32_RELATIVE_TOLERANCE
        ),
        "maximum_prediction_tolerance_ratio": tolerance_ratio,
        "passed": output["status"] == 0 and tolerance_ratio <= 1.0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
