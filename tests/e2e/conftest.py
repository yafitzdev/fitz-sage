# tests/e2e/conftest.py
"""
Pytest fixtures for E2E tests.

Provides the E2E runner as a module-scoped fixture that handles
setup (ingestion) and teardown (collection cleanup) automatically.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from .runner import E2ERunner, FIXTURES_DIR


@pytest.fixture(scope="module")
def e2e_runner():
    """
    Module-scoped E2E runner fixture.

    Sets up the test environment (ingests fixtures into a unique collection)
    before any tests run, and tears down (deletes collection) after all tests.

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
