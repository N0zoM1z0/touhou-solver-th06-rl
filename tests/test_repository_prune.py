from __future__ import annotations

import ast
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]

FORBIDDEN_PATH_PATTERNS = (
    "run_*learning.bat",
    "config/autonomous_generation[1-6]*.json",
    "docs/AUTONOMOUS*GENERATION_[1-6]*.md",
    "docs/GENERATION[1-6]*.md",
    "scripts/*generation[1-6]*.py",
    "scripts/*headless*.py",
    "scripts/*risk_guard*.py",
    "scripts/*risk_consensus*.py",
    "scripts/*retail*cow*.py",
    "scripts/*targeted*.py",
)

FORBIDDEN_PATHS = (
    "scripts/authorize_autonomous_canary.py",
    "scripts/authorize_conservative_canary.py",
    "scripts/authorize_option_advantage_canary.py",
    "scripts/authorize_sequential_r_canary.py",
    "scripts/authorize_supported_implicit_q_canary.py",
    "scripts/check_stage_mastery.py",
    "scripts/evaluate_learning.py",
    "scripts/replay_online_ucb.py",
    "scripts/sync_hf_corpus.py",
    "scripts/train_offline_cpu.py",
    "scripts/train_offline_fqi.py",
    "scripts/run_autonomous_learning.py",
    "scripts/run_autonomous_learning_v2.py",
    "scripts/run_autonomous_learning_v3.py",
    "scripts/run_autonomous_learning_v4.py",
    "scripts/run_autonomous_learning_v5.py",
    "src/th06_rl/advantage_learning.py",
    "src/th06_rl/audited_option_loader.py",
    "src/th06_rl/autonomous_learning.py",
    "src/th06_rl/conservative_learning.py",
    "src/th06_rl/curriculum_contract.py",
    "src/th06_rl/headless.py",
    "src/th06_rl/headless_corpus.py",
    "src/th06_rl/headless_forkserver.py",
    "src/th06_rl/headless_geometry.py",
    "src/th06_rl/implicit_learning.py",
    "src/th06_rl/iql_actor_learning.py",
    "src/th06_rl/low_rank_learning.py",
    "src/th06_rl/native_decision_conformance.py",
    "src/th06_rl/option_cache.py",
    "src/th06_rl/offline.py",
    "src/th06_rl/offline_learning.py",
    "src/th06_rl/offline_rl.py",
    "src/th06_rl/policies/adaptive.py",
    "src/th06_rl/policies/offline_risk_consensus.py",
    "src/th06_rl/policies/offline_risk_guard.py",
    "src/th06_rl/policies/autonomous_conservative_q.py",
    "src/th06_rl/policies/autonomous_dr_option_advantage.py",
    "src/th06_rl/policies/autonomous_iql_actor.py",
    "src/th06_rl/policies/autonomous_linear_q.py",
    "src/th06_rl/policies/autonomous_sequential_r_critic.py",
    "src/th06_rl/policies/autonomous_supported_implicit_q.py",
    "src/th06_rl/policies/generation6_collection_behavior.py",
    "src/th06_rl/policies/propensity_aware_option_exploration.py",
    "src/th06_rl/policy_transaction.py",
    "src/th06_rl/qualification_corpus.py",
    "src/th06_rl/sequential_learning.py",
    "src/th06_rl/wine_risk.py",
)

FORBIDDEN_MODULES = frozenset(
    {
        "headless",
        "headless_corpus",
        "headless_forkserver",
        "headless_geometry",
        "advantage_learning",
        "audited_option_loader",
        "autonomous_learning",
        "conservative_learning",
        "curriculum_contract",
        "implicit_learning",
        "iql_actor_learning",
        "low_rank_learning",
        "native_decision_conformance",
        "option_cache",
        "offline",
        "offline_learning",
        "offline_rl",
        "policy_transaction",
        "qualification_corpus",
        "sequential_learning",
        "wine_risk",
        "policies.adaptive",
        "policies.offline_risk_consensus",
        "policies.offline_risk_guard",
        "th06_rl.headless",
        "th06_rl.headless_corpus",
        "th06_rl.headless_forkserver",
        "th06_rl.headless_geometry",
        "th06_rl.advantage_learning",
        "th06_rl.audited_option_loader",
        "th06_rl.autonomous_learning",
        "th06_rl.conservative_learning",
        "th06_rl.curriculum_contract",
        "th06_rl.implicit_learning",
        "th06_rl.iql_actor_learning",
        "th06_rl.low_rank_learning",
        "th06_rl.native_decision_conformance",
        "th06_rl.option_cache",
        "th06_rl.offline",
        "th06_rl.offline_learning",
        "th06_rl.offline_rl",
        "th06_rl.policy_transaction",
        "th06_rl.qualification_corpus",
        "th06_rl.sequential_learning",
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


def test_retail_runtime_has_no_external_solver_import_boundary() -> None:
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
                    if module == "th06" or module.startswith("th06."):
                        violations.append(
                            f"{path.relative_to(REPOSITORY)}:{node.lineno}:{module}"
                        )

    assert not (REPOSITORY / "src/th06_rl/th06/donor.py").exists()
    assert violations == []


def test_portable_runtime_infra_has_no_home_directory_literals() -> None:
    roots = (
        REPOSITORY / "scripts",
        REPOSITORY / "src",
        REPOSITORY / "cmake",
        REPOSITORY / "config",
    )
    violations: list[str] = []
    for root in roots:
        for path in root.rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for literal in ("/home/", "/Users/"):
                if literal in content:
                    violations.append(
                        f"{path.relative_to(REPOSITORY)} contains {literal!r}"
                    )

    assert violations == []
