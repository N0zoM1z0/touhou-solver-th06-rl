from __future__ import annotations

import ast
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]

FORBIDDEN_PATH_PATTERNS = (
    "run_*learning.bat",
    "scripts/*headless*.py",
    "scripts/*risk_guard*.py",
    "scripts/*risk_consensus*.py",
    "scripts/*retail*cow*.py",
    "scripts/*targeted*.py",
)

FORBIDDEN_PATHS = (
    "scripts/check_stage_mastery.py",
    "scripts/evaluate_learning.py",
    "scripts/replay_online_ucb.py",
    "scripts/sync_hf_corpus.py",
    "scripts/train_offline_cpu.py",
    "scripts/train_offline_fqi.py",
    "src/th06_rl/headless.py",
    "src/th06_rl/headless_corpus.py",
    "src/th06_rl/headless_forkserver.py",
    "src/th06_rl/headless_geometry.py",
    "src/th06_rl/offline.py",
    "src/th06_rl/offline_learning.py",
    "src/th06_rl/offline_rl.py",
    "src/th06_rl/policies/adaptive.py",
    "src/th06_rl/policies/offline_risk_consensus.py",
    "src/th06_rl/policies/offline_risk_guard.py",
    "src/th06_rl/policy_transaction.py",
    "src/th06_rl/wine_risk.py",
)

FORBIDDEN_MODULES = frozenset(
    {
        "headless",
        "headless_corpus",
        "headless_forkserver",
        "headless_geometry",
        "offline",
        "offline_learning",
        "offline_rl",
        "policy_transaction",
        "wine_risk",
        "policies.adaptive",
        "policies.offline_risk_consensus",
        "policies.offline_risk_guard",
        "th06_rl.headless",
        "th06_rl.headless_corpus",
        "th06_rl.headless_forkserver",
        "th06_rl.headless_geometry",
        "th06_rl.offline",
        "th06_rl.offline_learning",
        "th06_rl.offline_rl",
        "th06_rl.policy_transaction",
        "th06_rl.wine_risk",
        "th06_rl.policies.adaptive",
        "th06_rl.policies.offline_risk_consensus",
        "th06_rl.policies.offline_risk_guard",
    }
)


def test_retired_learning_backends_remain_absent() -> None:
    forbidden = {
        path
        for pattern in FORBIDDEN_PATH_PATTERNS
        for path in REPOSITORY.glob(pattern)
    }
    forbidden.update(
        path for name in FORBIDDEN_PATHS if (path := REPOSITORY / name).exists()
    )

    assert forbidden == set()


def test_retained_python_does_not_import_retired_backends() -> None:
    roots = (REPOSITORY / "src", REPOSITORY / "scripts", REPOSITORY / "tests")
    violations: list[str] = []
    for root in roots:
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules = tuple(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module is not None:
                    modules = (node.module,)
                else:
                    continue
                for module in modules:
                    if module in FORBIDDEN_MODULES:
                        violations.append(
                            f"{path.relative_to(REPOSITORY)}:{node.lineno}:{module}"
                        )

    assert violations == []
