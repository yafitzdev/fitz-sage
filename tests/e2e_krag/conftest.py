# tests/e2e_krag/conftest.py
"""
Pytest fixtures for KRAG E2E tests.

Equivalent to tests/e2e/conftest.py but uses KragE2ERunner
instead of E2ERunner.

Configuration:
- All tests use tests/test_config.yaml (same as unit, load, etc.)
- FitzPaths workspace is set to project root for vocabulary/entity graph storage

Tiered Execution:
- By default, runs all tests through tiered fallback (local -> cloud)
- Results are cached so individual pytest tests just look up pre-computed results
- Set E2E_SINGLE_TIER=1 to disable tiered mode and use only the first tier
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import pytest

from tests.e2e.scenarios import SCENARIOS, TestScenario

from .runner import FIXTURES_DIR, KragE2ERunner


def pytest_collection_modifyitems(items):
    """Add tier3 and e2e_krag markers to all tests in this directory."""
    for item in items:
        if "/e2e_krag/" in str(item.fspath) or "\\e2e_krag\\" in str(item.fspath):
            item.add_marker(pytest.mark.tier3)
            item.add_marker(pytest.mark.e2e_krag)


# Set workspace to project root so FitzPaths finds storage paths
PROJECT_ROOT = Path(__file__).parent.parent.parent


@pytest.fixture(scope="session", autouse=True)
def set_workspace():
    """Set FitzPaths workspace to project root before any tests run."""
    from fitz_ai.core.paths import FitzPaths

    FitzPaths.set_workspace(PROJECT_ROOT / ".fitz")
    yield
    FitzPaths.reset()


def create_tiered_krag_runner(
    fixtures_dir: Path,
    scenarios: list[TestScenario],
    suite_name: str = "KRAG E2E",
) -> Callable[[], KragE2ERunner]:
    """
    Factory to create tiered KRAG E2E runner setup logic.

    Args:
        fixtures_dir: Path to test fixtures
        scenarios: List of test scenarios to run
        suite_name: Name for the test suite (shown in header)

    Returns:
        Generator function that yields the configured runner
    """

    def runner_generator():
        runner = KragE2ERunner(fixtures_dir=fixtures_dir, use_cache=True)
        runner.setup()

        use_tiered = os.environ.get("E2E_SINGLE_TIER", "0") != "1"

        if use_tiered:
            print("\n" + "=" * 60)
            print(f"{suite_name} TESTS - TIERED EXECUTION")
            print("(local -> cloud fallback)")
            print("Set E2E_SINGLE_TIER=1 to disable")
            print("=" * 60 + "\n")

            runner._tiered_results = runner.run_tiered(scenarios)
        else:
            runner._tiered_results = None

        yield runner
        runner.teardown()

    return runner_generator


@pytest.fixture(scope="session")
def krag_e2e_runner(set_workspace):
    """
    Session-scoped KRAG E2E runner fixture for main retrieval tests.

    By default, runs ALL scenarios through tiered execution (local -> cloud)
    during setup. Individual tests then just look up their pre-computed result
    via runner.get_tiered_result(scenario_id).

    Set E2E_SINGLE_TIER=1 to disable tiered mode.
    """
    yield from create_tiered_krag_runner(FIXTURES_DIR, SCENARIOS, "MAIN KRAG E2E")()


@pytest.fixture(scope="module")
def fixtures_path() -> Path:
    """Return the path to E2E test fixtures."""
    return FIXTURES_DIR
