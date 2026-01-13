# tests/e2e/conftest.py
"""
Pytest fixtures for E2E tests.

Provides the E2E runner as a module-scoped fixture that handles
setup (ingestion) and teardown (collection cleanup) automatically.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from .runner import FIXTURES_DIR, E2ERunner

# Set workspace to project root so FitzPaths finds .fitz/config/
# This is needed because PyCharm may run pytest from a different CWD
PROJECT_ROOT = Path(__file__).parent.parent.parent


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

    Sets up the test environment once per pytest session. The setup() call
    automatically cleans up any stale collections via _cleanup_stale_data().

    Usage:
        def test_something(e2e_runner):
            result = e2e_runner.run_scenario(scenario)
            assert result.validation.passed
    """
    runner = E2ERunner(fixtures_dir=FIXTURES_DIR)
    runner.setup()
    yield runner
    runner.teardown()


@pytest.fixture(scope="module")
def fixtures_path() -> Path:
    """Return the path to E2E test fixtures."""
    return FIXTURES_DIR
