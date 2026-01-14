# tests/e2e/conftest.py
"""
Pytest fixtures for E2E tests.

Provides the E2E runner as a session-scoped fixture that handles
setup (ingestion) and teardown (collection cleanup) automatically.

Configuration:
- LLM/embedding config is loaded from tests/e2e/e2e_config.yaml
- FitzPaths workspace is set to project root for vocabulary/entity graph storage

Tiered Execution:
- By default, runs all tests through tiered fallback (local -> cloud)
- Results are cached so individual pytest tests just look up pre-computed results
- Set E2E_SINGLE_TIER=1 to disable tiered mode and use only the first tier
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from .runner import FIXTURES_DIR, E2ERunner, TieredRunResult
from .scenarios import SCENARIOS

# Set workspace to project root so FitzPaths finds storage paths
# (vocabulary, entity graph, table cache, etc.)
# Note: LLM config is now loaded from tests/e2e/e2e_config.yaml, not .fitz/config/
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Global storage for tiered results (populated once per session)
_tiered_results: TieredRunResult | None = None


@pytest.fixture(scope="session", autouse=True)
def set_workspace():
    """Set FitzPaths workspace to project root before any tests run."""
    from fitz_ai.core.paths import FitzPaths

    FitzPaths.set_workspace(PROJECT_ROOT / ".fitz")
    yield
    FitzPaths.reset()


@pytest.fixture(scope="session")
def e2e_runner(set_workspace):
    """
    Session-scoped E2E runner fixture.

    By default, runs ALL scenarios through tiered execution (local -> cloud)
    during setup. Individual tests then just look up their pre-computed result.

    Set E2E_SINGLE_TIER=1 to disable tiered mode.

    Usage:
        def test_something(e2e_runner):
            result = e2e_runner.run_scenario(scenario)
            assert result.validation.passed
    """
    global _tiered_results

    runner = E2ERunner(fixtures_dir=FIXTURES_DIR)
    runner.setup()

    # Run tiered execution unless disabled
    use_tiered = os.environ.get("E2E_SINGLE_TIER", "0") != "1"

    if use_tiered:
        print("\n" + "=" * 60)
        print("TIERED E2E EXECUTION (local -> cloud fallback)")
        print("Set E2E_SINGLE_TIER=1 to disable")
        print("=" * 60 + "\n")

        _tiered_results = runner.run_tiered(SCENARIOS)

        # Store results on runner for lookup
        runner._tiered_results = _tiered_results
    else:
        runner._tiered_results = None

    yield runner
    runner.teardown()


@pytest.fixture(scope="module")
def fixtures_path() -> Path:
    """Return the path to E2E test fixtures."""
    return FIXTURES_DIR


def get_tiered_result(scenario_id: str):
    """Get pre-computed result for a scenario from tiered execution."""
    global _tiered_results
    if _tiered_results is None:
        return None
    return _tiered_results.results.get(scenario_id)
