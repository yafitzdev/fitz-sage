# tests/security/conftest.py
"""
Security test fixtures with tiered fallback support.

Security tests run with local LLM first, then retry with cloud on failure.
"""

from __future__ import annotations

import functools
from typing import Callable

import pytest

# Mark all tests in this directory as tier4 and security
def pytest_configure(config):
    """Apply markers to all tests in this directory."""
    pass


def pytest_collection_modifyitems(items):
    """Add tier4 and security markers to all tests in this directory."""
    for item in items:
        if "/security/" in str(item.fspath) or "\\security\\" in str(item.fspath):
            item.add_marker(pytest.mark.tier4)
            item.add_marker(pytest.mark.security)

from tests.e2e.config import get_tier_names, load_test_config


def with_tiered_fallback(test_fn: Callable) -> Callable:
    """
    Decorator that retries a test with cloud tier if local tier fails.

    Usage:
        @with_tiered_fallback
        def test_something(self):
            result = self.runner.pipeline.run("query")
            assert "expected" in result.answer
    """

    @functools.wraps(test_fn)
    def wrapper(self, *args, **kwargs):
        config = load_test_config()
        tier_names = get_tier_names(config)

        last_error = None
        for tier_name in tier_names:
            try:
                # Switch to this tier
                self.runner._rebuild_pipeline(tier_name)

                # Run the test
                test_fn(self, *args, **kwargs)

                # Success - return
                return
            except (AssertionError, RuntimeError) as e:
                last_error = e
                # Continue to next tier

        # All tiers failed
        raise last_error

    return wrapper


@pytest.fixture
def tiered_pipeline(e2e_runner):
    """
    Fixture providing a pipeline that automatically retries with cloud tier.

    Returns a callable that runs a query and assertion with tiered fallback.
    """
    config = load_test_config()
    tier_names = get_tier_names(config)

    def run_with_fallback(query: str, assertion: Callable[[str], bool], error_msg: str = ""):
        """
        Run query through pipeline, retrying with cloud tier if assertion fails.

        Args:
            query: The query to run
            assertion: Function taking answer string, returns True if passed
            error_msg: Message to show on failure

        Returns:
            The answer text from the passing tier

        Raises:
            AssertionError: If all tiers fail the assertion
        """
        last_error = None
        last_answer = ""

        for tier_name in tier_names:
            try:
                e2e_runner._rebuild_pipeline(tier_name)
                result = e2e_runner.pipeline.run(query)
                last_answer = result.answer

                if assertion(result.answer):
                    return result.answer
                else:
                    last_error = AssertionError(
                        f"{error_msg or 'Assertion failed'} (tier={tier_name}): {last_answer[:200]}"
                    )
            except RuntimeError as e:
                last_error = e

        raise last_error or AssertionError(f"All tiers failed: {last_answer[:200]}")

    return run_with_fallback
